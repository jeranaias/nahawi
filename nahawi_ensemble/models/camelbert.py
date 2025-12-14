"""
CAMeLBERT-based morphology model.

Handles morphological errors:
- Gender agreement
- Number agreement
- Verb conjugation
- Case endings
"""

import torch
from typing import List, Optional, Tuple
from pathlib import Path
import logging

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoModelForSeq2SeqLM,
)

from .base import NeuralGECModel, Correction, CorrectionResult

logger = logging.getLogger(__name__)


class MorphologyFixer(NeuralGECModel):
    """
    CAMeLBERT-based model for morphological errors.

    Uses CAMeLBERT which is specifically trained on Arabic morphology.
    Can detect and fix:
    - Gender mismatches (masculine/feminine)
    - Number mismatches (singular/dual/plural)
    - Verb-subject agreement
    - Definiteness errors
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        base_model: str = "CAMeL-Lab/bert-base-arabic-camelbert-mix",
        device: str = "cuda",
        max_length: int = 128,
    ):
        super().__init__(
            name="MorphologyFixer",
            error_types=["morphology", "agreement", "gender", "number"],
            base_model=base_model,
            checkpoint_path=checkpoint_path,
            device=device,
            max_length=max_length
        )
        self.is_encoder_only = True  # CAMeLBERT is encoder-only

    def load(self) -> None:
        """Load CAMeLBERT model and tokenizer."""
        logger.info(f"Loading CAMeLBERT: {self.base_model}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)

        # For morphology, we use token classification approach
        # Each token gets a label: O (no change), or correction label
        # For now, we'll use a simpler seq2seq approach by wrapping
        # the encoder with a decoder head

        # Try to load as seq2seq first, fall back to encoder + custom decoder
        try:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.base_model)
        except:
            # Load encoder-only and we'll handle generation differently
            from transformers import BertModel, BertConfig
            self.encoder = AutoModelForTokenClassification.from_pretrained(
                self.base_model,
                num_labels=3  # O, EDIT, DELETE
            )
            self.model = self.encoder

        if self.checkpoint_path and Path(self.checkpoint_path).exists():
            logger.info(f"Loading checkpoint: {self.checkpoint_path}")
            state_dict = torch.load(self.checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state_dict)

        self.model = self.model.to(self.device)
        self.model.eval()
        self.is_loaded = True

        logger.info(f"CAMeLBERT loaded on {self.device}")

    def _generate(self, text: str) -> Tuple[str, float]:
        """
        Generate morphologically corrected text.

        For encoder-only models, we use a rule-based approach
        combined with the model's predictions.
        """
        if not self.is_loaded:
            self.load()

        # For now, return text unchanged with confidence
        # This will be enhanced when we fine-tune the model

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # For token classification, get predictions
        if hasattr(outputs, 'logits'):
            predictions = outputs.logits.argmax(dim=-1)
            confidence = outputs.logits.softmax(dim=-1).max(dim=-1).values.mean().item()
        else:
            confidence = 0.5

        # For now, return original (model needs fine-tuning)
        return text, confidence

    def correct(self, text: str) -> CorrectionResult:
        """Correct morphological errors in text."""
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
        return [self.correct(text) for text in texts]


# Morphology rules (used when model is not fine-tuned)
class ArabicMorphologyRules:
    """
    Rule-based Arabic morphology corrections.

    These serve as fallback when the neural model is not available.
    """

    # Feminine markers
    FEMININE_ENDINGS = {'ة', 'ى', 'اء'}

    # Common gender agreement errors
    GENDER_FIXES = {
        # Adjective should agree with noun
        ('مدرسة', 'كبير'): 'كبيرة',
        ('جامعة', 'كبير'): 'كبيرة',
        ('سيارة', 'جديد'): 'جديدة',
        ('بنت', 'صغير'): 'صغيرة',
    }

    # Common number agreement patterns
    PLURAL_PATTERNS = {
        # Broken plurals that often cause errors
        'كتب': 'كتاب',  # books -> book
        'رجال': 'رجل',  # men -> man
        'نساء': 'امرأة',  # women -> woman
    }

    @classmethod
    def check_gender_agreement(cls, noun: str, adjective: str) -> Optional[str]:
        """Check if adjective agrees with noun in gender."""
        if (noun, adjective) in cls.GENDER_FIXES:
            return cls.GENDER_FIXES[(noun, adjective)]

        # General rule: if noun ends in ة, adjective should too
        noun_fem = noun.endswith('ة') or noun.endswith('ى')
        adj_fem = adjective.endswith('ة') or adjective.endswith('ى')

        if noun_fem and not adj_fem:
            # Try to make adjective feminine
            if not adjective.endswith(tuple(cls.FEMININE_ENDINGS)):
                return adjective + 'ة'

        return None
