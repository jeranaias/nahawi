"""
AraBART-based GEC models.

This provides the base for:
- HamzaFixer
- SpaceFixer
- DeletedWordFixer
- SpellingFixer
- GeneralGEC
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional, Tuple, Dict
from pathlib import Path
import logging

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)

from .base import NeuralGECModel, Correction, CorrectionResult

logger = logging.getLogger(__name__)


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')

    try:
        import torch_directml
        if torch_directml.is_available():
            return torch_directml.device()
    except ImportError:
        pass

    return torch.device('cpu')


class GECDataset(Dataset):
    """Dataset for GEC training."""

    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        tokenizer,
        max_length: int = 128
    ):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        source, target = self.pairs[idx]

        source_encoding = self.tokenizer(
            source,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        target_encoding = self.tokenizer(
            target,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'input_ids': source_encoding['input_ids'].squeeze(),
            'attention_mask': source_encoding['attention_mask'].squeeze(),
            'labels': target_encoding['input_ids'].squeeze(),
        }


class AraBART_GEC(NeuralGECModel):
    """
    AraBART-based GEC model.

    Can be specialized for different error types by training on
    focused datasets.
    """

    def __init__(
        self,
        name: str,
        error_types: List[str],
        checkpoint_path: Optional[str] = None,
        base_model: str = "moussaKam/AraBART",
        device: str = "cuda",
        max_length: int = 128,
    ):
        super().__init__(
            name=name,
            error_types=error_types,
            base_model=base_model,
            checkpoint_path=checkpoint_path,
            device=device,
            max_length=max_length
        )

    def load(self) -> None:
        """Load AraBART model and tokenizer."""
        logger.info(f"Loading AraBART model: {self.base_model}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.base_model)

        if self.checkpoint_path and Path(self.checkpoint_path).exists():
            logger.info(f"Loading checkpoint: {self.checkpoint_path}")
            state_dict = torch.load(self.checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state_dict)

        self.model = self.model.to(self.device)
        self.model.eval()
        self.is_loaded = True

        logger.info(f"Model loaded on {self.device}")

    def train(
        self,
        train_pairs: List[Tuple[str, str]],
        val_pairs: Optional[List[Tuple[str, str]]] = None,
        output_dir: str = "./checkpoints",
        epochs: int = 5,
        batch_size: int = 8,
        learning_rate: float = 2e-5,
        warmup_ratio: float = 0.1,
        gradient_accumulation_steps: int = 4,
        fp16: bool = True,
    ) -> None:
        """
        Fine-tune the model on GEC data.

        Args:
            train_pairs: List of (error, correct) pairs
            val_pairs: Optional validation pairs
            output_dir: Where to save checkpoints
            epochs: Number of training epochs
            batch_size: Batch size per device
            learning_rate: Learning rate
            warmup_ratio: Warmup ratio
            gradient_accumulation_steps: Gradient accumulation
            fp16: Use mixed precision
        """
        if not self.is_loaded:
            self.load()

        logger.info(f"Training {self.name} on {len(train_pairs)} pairs")

        # Create datasets
        train_dataset = GECDataset(train_pairs, self.tokenizer, self.max_length)

        eval_dataset = None
        if val_pairs:
            eval_dataset = GECDataset(val_pairs, self.tokenizer, self.max_length)

        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True,
        )

        # Training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_ratio=warmup_ratio,
            gradient_accumulation_steps=gradient_accumulation_steps,
            fp16=fp16 and self.device != "cpu",
            evaluation_strategy="epoch" if eval_dataset else "no",
            save_strategy="epoch",
            logging_steps=100,
            save_total_limit=2,
            predict_with_generate=True,
            load_best_model_at_end=True if eval_dataset else False,
            report_to="none",
        )

        # Trainer
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )

        # Train
        trainer.train()

        # Save final model
        final_path = Path(output_dir) / "final"
        trainer.save_model(str(final_path))
        self.checkpoint_path = str(final_path / "pytorch_model.bin")

        logger.info(f"Training complete. Model saved to {final_path}")

    def _generate(self, text: str) -> Tuple[str, float]:
        """Generate corrected text."""
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
                no_repeat_ngram_size=3,
            )

        corrected = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Confidence estimation (simplified)
        # In practice, you'd want beam scores or similar
        confidence = 0.8 if corrected != text else 0.5

        return corrected, confidence

    def correct_batch(self, texts: List[str]) -> List[CorrectionResult]:
        """Correct a batch of texts efficiently."""
        if not self.is_loaded:
            self.load()

        results = []

        # Batch tokenization
        inputs = self.tokenizer(
            texts,
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
                no_repeat_ngram_size=3,
            )

        for i, (text, output) in enumerate(zip(texts, outputs)):
            corrected = self.tokenizer.decode(output, skip_special_tokens=True)
            corrections = self._find_corrections(text, corrected)

            results.append(CorrectionResult(
                original_text=text,
                corrected_text=corrected,
                corrections=corrections,
                confidence=0.8 if corrections else 0.5
            ))

        return results


class HamzaFixer(AraBART_GEC):
    """Specialized model for Hamza errors."""

    def __init__(self, checkpoint_path: Optional[str] = None, device: str = "cuda"):
        super().__init__(
            name="HamzaFixer",
            error_types=["hamza_alif", "hamza_waw", "hamza_ya", "hamza_alone"],
            checkpoint_path=checkpoint_path,
            device=device
        )


class SpaceFixer(AraBART_GEC):
    """Specialized model for space/merge/split errors."""

    def __init__(self, checkpoint_path: Optional[str] = None, device: str = "cuda"):
        super().__init__(
            name="SpaceFixer",
            error_types=["merge", "split", "add_space"],
            checkpoint_path=checkpoint_path,
            device=device
        )


class DeletedWordFixer(AraBART_GEC):
    """Specialized model for missing words."""

    def __init__(self, checkpoint_path: Optional[str] = None, device: str = "cuda"):
        super().__init__(
            name="DeletedWordFixer",
            error_types=["deleted_word"],
            checkpoint_path=checkpoint_path,
            device=device
        )


class SpellingFixer(AraBART_GEC):
    """Specialized model for spelling errors."""

    def __init__(self, checkpoint_path: Optional[str] = None, device: str = "cuda"):
        super().__init__(
            name="SpellingFixer",
            error_types=["spelling", "char_swap", "char_delete", "char_insert"],
            checkpoint_path=checkpoint_path,
            device=device
        )


class GeneralGEC(AraBART_GEC):
    """General catch-all GEC model."""

    def __init__(self, checkpoint_path: Optional[str] = None, device: str = "cuda"):
        super().__init__(
            name="GeneralGEC",
            error_types=["general"],
            checkpoint_path=checkpoint_path,
            device=device
        )


# Factory function
def get_arabart_model(model_type: str, checkpoint_path: Optional[str] = None, device: str = "cuda") -> AraBART_GEC:
    """Get an AraBART-based model by type."""
    models = {
        'hamza': HamzaFixer,
        'space': SpaceFixer,
        'deleted_word': DeletedWordFixer,
        'spelling': SpellingFixer,
        'general': GeneralGEC,
    }

    if model_type not in models:
        raise ValueError(f"Unknown AraBART model: {model_type}")

    return models[model_type](checkpoint_path=checkpoint_path, device=device)
