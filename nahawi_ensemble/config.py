"""
Configuration for Nahawi Ensemble
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import torch


@dataclass
class ModelConfig:
    """Configuration for a single model in the ensemble."""
    name: str
    model_type: str  # 'arabart', 'camelbert', 'rule_based', 'classifier'
    base_model: str  # HuggingFace model name or 'none' for rule-based
    checkpoint_path: Optional[Path] = None
    enabled: bool = True
    priority: int = 5  # 1-10, higher = more trusted for its specialty
    error_types: List[str] = field(default_factory=list)  # What errors this model handles
    confidence_threshold: float = 0.8  # Minimum confidence to apply correction


@dataclass
class Config:
    """Main configuration for the ensemble."""

    # Paths
    base_dir: Path = Path("C:/nahawi")
    models_dir: Path = Path("C:/nahawi/nahawi_ensemble/checkpoints")
    data_dir: Path = Path("C:/nahawi/qalb_real_data")
    wiki_path: Path = Path("C:/nahawi/arabic_wiki/sentences.txt")
    patterns_path: Path = Path("C:/nahawi/qalb_correct_to_errors.json")

    # Base models (HuggingFace)
    arabart_model: str = "moussaKam/AraBART"
    camelbert_model: str = "CAMeL-Lab/bert-base-arabic-camelbert-mix"
    arabert_model: str = "aubmindlab/bert-base-arabertv2"

    # Training
    batch_size: int = 8
    learning_rate: float = 2e-5
    epochs: int = 5
    warmup_ratio: float = 0.1
    max_length: int = 128
    gradient_accumulation_steps: int = 4

    # Inference
    inference_batch_size: int = 16
    use_fp16: bool = True

    # Device
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    # Ensemble models
    models: Dict[str, ModelConfig] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize model configurations."""
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Define all 10 models
        self.models = {
            "hamza_fixer": ModelConfig(
                name="HamzaFixer",
                model_type="arabart",
                base_model=self.arabart_model,
                priority=10,
                error_types=["hamza_alif", "hamza_waw", "hamza_ya", "hamza_alone"],
                confidence_threshold=0.85,
            ),
            "space_fixer": ModelConfig(
                name="SpaceFixer",
                model_type="arabart",
                base_model=self.arabart_model,
                priority=9,
                error_types=["merge", "split", "add_space"],
                confidence_threshold=0.85,
            ),
            "taa_marbuta_fixer": ModelConfig(
                name="TaaMarbutaFixer",
                model_type="rule_based",
                base_model="none",
                priority=8,
                error_types=["taa_marbuta"],
                confidence_threshold=0.95,
            ),
            "alif_maksura_fixer": ModelConfig(
                name="AlifMaksuraFixer",
                model_type="rule_based",
                base_model="none",
                priority=8,
                error_types=["alif_maksura"],
                confidence_threshold=0.95,
            ),
            "punctuation_fixer": ModelConfig(
                name="PunctuationFixer",
                model_type="rule_based",
                base_model="none",
                priority=3,
                error_types=["punctuation"],
                confidence_threshold=0.99,
            ),
            "deleted_word_fixer": ModelConfig(
                name="DeletedWordFixer",
                model_type="arabart",
                base_model=self.arabart_model,
                priority=7,
                error_types=["deleted_word"],
                confidence_threshold=0.80,
            ),
            "repeated_word_fixer": ModelConfig(
                name="RepeatedWordFixer",
                model_type="rule_based",
                base_model="none",
                priority=9,
                error_types=["repeated_word"],
                confidence_threshold=0.99,
            ),
            "spelling_fixer": ModelConfig(
                name="SpellingFixer",
                model_type="arabart",
                base_model=self.arabart_model,
                priority=6,
                error_types=["spelling", "char_swap", "char_delete", "char_insert"],
                confidence_threshold=0.75,
            ),
            "morphology_fixer": ModelConfig(
                name="MorphologyFixer",
                model_type="camelbert",
                base_model=self.camelbert_model,
                priority=5,
                error_types=["morphology", "agreement", "gender", "number"],
                confidence_threshold=0.80,
            ),
            "general_gec": ModelConfig(
                name="GeneralGEC",
                model_type="arabart",
                base_model=self.arabart_model,
                priority=4,  # Lower priority - catch-all
                error_types=["general"],
                confidence_threshold=0.70,
            ),
        }

    def get_enabled_models(self) -> Dict[str, ModelConfig]:
        """Get only enabled models."""
        return {k: v for k, v in self.models.items() if v.enabled}

    def get_models_by_priority(self) -> List[tuple]:
        """Get models sorted by priority (highest first)."""
        return sorted(
            self.models.items(),
            key=lambda x: x[1].priority,
            reverse=True
        )


# Global config instance
config = Config()
