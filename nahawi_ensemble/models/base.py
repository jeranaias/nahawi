"""
Base classes for GEC models in the ensemble.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple
import torch


@dataclass
class Correction:
    """Represents a single correction."""
    original: str           # Original text span
    corrected: str          # Corrected text span
    start_idx: int          # Character start position in sentence
    end_idx: int            # Character end position in sentence
    error_type: str         # Type of error (hamza, space, etc.)
    confidence: float       # Model confidence (0-1)
    model_name: str         # Which model suggested this


@dataclass
class CorrectionResult:
    """Result from a model's correction attempt."""
    original_text: str
    corrected_text: str
    corrections: List[Correction]
    confidence: float  # Overall confidence


class BaseGECModel(ABC):
    """
    Abstract base class for all GEC models in the ensemble.

    All models must implement:
    - correct(): Takes text, returns corrected text + corrections list
    - get_corrections(): Takes text, returns only corrections (no application)
    """

    def __init__(self, name: str, error_types: List[str], device: str = "cuda"):
        self.name = name
        self.error_types = error_types
        self.device = device
        self.is_loaded = False

    @abstractmethod
    def load(self) -> None:
        """Load model weights/resources."""
        pass

    @abstractmethod
    def correct(self, text: str) -> CorrectionResult:
        """
        Correct the input text.

        Args:
            text: Input text (possibly with errors)

        Returns:
            CorrectionResult with corrected text and list of corrections
        """
        pass

    @abstractmethod
    def correct_batch(self, texts: List[str]) -> List[CorrectionResult]:
        """
        Correct a batch of texts.

        Args:
            texts: List of input texts

        Returns:
            List of CorrectionResults
        """
        pass

    def get_corrections(self, text: str) -> List[Correction]:
        """
        Get corrections without applying them.

        Default implementation calls correct() and extracts corrections.
        Override for efficiency if needed.
        """
        result = self.correct(text)
        return result.corrections

    def handles_error_type(self, error_type: str) -> bool:
        """Check if this model handles a specific error type."""
        return error_type in self.error_types or "general" in self.error_types

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}', types={self.error_types})"


class NeuralGECModel(BaseGECModel):
    """
    Base class for neural (transformer-based) GEC models.

    Provides common functionality for AraBART/CAMeLBERT models.
    """

    def __init__(
        self,
        name: str,
        error_types: List[str],
        base_model: str,
        checkpoint_path: Optional[str] = None,
        device: str = "cuda",
        max_length: int = 128,
    ):
        super().__init__(name, error_types, device)
        self.base_model = base_model
        self.checkpoint_path = checkpoint_path
        self.max_length = max_length
        self.model = None
        self.tokenizer = None

    def load(self) -> None:
        """Load the transformer model and tokenizer."""
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.base_model)

        if self.checkpoint_path:
            # Load fine-tuned weights
            state_dict = torch.load(self.checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state_dict)

        self.model = self.model.to(self.device)
        self.model.eval()
        self.is_loaded = True

    def _generate(self, text: str) -> Tuple[str, float]:
        """
        Generate corrected text using the model.

        Returns:
            Tuple of (corrected_text, confidence)
        """
        if not self.is_loaded:
            self.load()

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=self.max_length,
                num_beams=4,
                early_stopping=True,
                return_dict_in_generate=True,
                output_scores=True,
            )

        corrected = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

        # Calculate confidence from sequence scores
        if hasattr(outputs, 'sequences_scores') and outputs.sequences_scores is not None:
            confidence = torch.sigmoid(outputs.sequences_scores[0]).item()
        else:
            confidence = 0.5  # Default confidence

        return corrected, confidence

    def _find_corrections(
        self,
        original: str,
        corrected: str
    ) -> List[Correction]:
        """
        Find the differences between original and corrected text.

        Simple word-level diff. Override for more sophisticated alignment.
        """
        corrections = []

        orig_words = original.split()
        corr_words = corrected.split()

        # Simple alignment - works for most cases
        i, j = 0, 0
        char_pos = 0

        while i < len(orig_words) or j < len(corr_words):
            if i >= len(orig_words):
                # Insertion in corrected
                corrections.append(Correction(
                    original="",
                    corrected=corr_words[j],
                    start_idx=char_pos,
                    end_idx=char_pos,
                    error_type="insertion",
                    confidence=0.8,
                    model_name=self.name
                ))
                j += 1
            elif j >= len(corr_words):
                # Deletion in corrected
                word_len = len(orig_words[i])
                corrections.append(Correction(
                    original=orig_words[i],
                    corrected="",
                    start_idx=char_pos,
                    end_idx=char_pos + word_len,
                    error_type="deletion",
                    confidence=0.8,
                    model_name=self.name
                ))
                char_pos += word_len + 1
                i += 1
            elif orig_words[i] == corr_words[j]:
                # Match
                char_pos += len(orig_words[i]) + 1
                i += 1
                j += 1
            else:
                # Substitution
                word_len = len(orig_words[i])
                corrections.append(Correction(
                    original=orig_words[i],
                    corrected=corr_words[j],
                    start_idx=char_pos,
                    end_idx=char_pos + word_len,
                    error_type=self._classify_error(orig_words[i], corr_words[j]),
                    confidence=0.8,
                    model_name=self.name
                ))
                char_pos += word_len + 1
                i += 1
                j += 1

        return corrections

    def _classify_error(self, original: str, corrected: str) -> str:
        """Classify the type of error based on the change."""
        # Hamza errors
        hamza_chars = set('أإآءؤئا')
        if any(c in original for c in hamza_chars) or any(c in corrected for c in hamza_chars):
            if set(original) - hamza_chars == set(corrected) - hamza_chars:
                return "hamza"

        # Taa marbuta
        if original.endswith('ه') and corrected.endswith('ة'):
            return "taa_marbuta"
        if original.endswith('ة') and corrected.endswith('ه'):
            return "taa_marbuta"

        # Alif maksura
        if original.endswith('ي') and corrected.endswith('ى'):
            return "alif_maksura"
        if original.endswith('ى') and corrected.endswith('ي'):
            return "alif_maksura"

        # Length difference suggests spelling
        if abs(len(original) - len(corrected)) <= 2:
            return "spelling"

        return "general"

    def correct(self, text: str) -> CorrectionResult:
        """Correct the input text."""
        corrected, confidence = self._generate(text)
        corrections = self._find_corrections(text, corrected)

        return CorrectionResult(
            original_text=text,
            corrected_text=corrected,
            corrections=corrections,
            confidence=confidence
        )

    def correct_batch(self, texts: List[str]) -> List[CorrectionResult]:
        """Correct a batch of texts."""
        if not self.is_loaded:
            self.load()

        results = []
        for text in texts:
            results.append(self.correct(text))

        return results


class RuleBasedModel(BaseGECModel):
    """
    Base class for rule-based models.

    These models use linguistic rules rather than neural networks.
    They're fast and high-precision for specific error types.
    """

    def __init__(self, name: str, error_types: List[str]):
        super().__init__(name, error_types, device="cpu")
        self.is_loaded = True  # Rule-based models are always "loaded"

    def load(self) -> None:
        """No loading needed for rule-based models."""
        pass

    @abstractmethod
    def apply_rules(self, text: str) -> Tuple[str, List[Correction]]:
        """
        Apply correction rules to the text.

        Returns:
            Tuple of (corrected_text, corrections_list)
        """
        pass

    def correct(self, text: str) -> CorrectionResult:
        """Correct the input text using rules."""
        corrected, corrections = self.apply_rules(text)

        # Rule-based corrections are high confidence
        confidence = 0.95 if corrections else 1.0

        return CorrectionResult(
            original_text=text,
            corrected_text=corrected,
            corrections=corrections,
            confidence=confidence
        )

    def correct_batch(self, texts: List[str]) -> List[CorrectionResult]:
        """Correct a batch of texts."""
        return [self.correct(text) for text in texts]
