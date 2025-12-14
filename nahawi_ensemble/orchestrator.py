"""
Nahawi Ensemble Orchestrator
============================

Coordinates multiple GEC models to produce the best corrections.

Strategies:
1. Cascading: Fast rule-based models first, then neural
2. Specialist routing: Route specific errors to specialist models
3. Confidence-based merging: Trust high-confidence corrections
"""

import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from .config import Config, config
from .models.base import BaseGECModel, Correction, CorrectionResult
from .models.rule_based import (
    TaaMarbutaFixer,
    AlifMaksuraFixer,
    PunctuationFixer,
    RepeatedWordFixer,
)
from .models.arabart_base import (
    HamzaFixer,
    SpaceFixer,
    DeletedWordFixer,
    SpellingFixer,
    GeneralGEC,
)

logger = logging.getLogger(__name__)


@dataclass
class EnsembleResult:
    """Result from the ensemble."""
    original_text: str
    corrected_text: str
    corrections: List[Correction]
    model_contributions: Dict[str, int]  # Which model made how many corrections
    confidence: float


class NahawiEnsemble:
    """
    Main ensemble orchestrator.

    Coordinates all 10 models to produce corrections.
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        lazy_load: bool = True,
        enabled_models: Optional[List[str]] = None,
    ):
        """
        Initialize the ensemble.

        Args:
            config: Configuration object
            lazy_load: If True, only load models when first needed
            enabled_models: List of model names to enable (None = all)
        """
        self.config = config or Config()
        self.lazy_load = lazy_load
        self.enabled_models = enabled_models

        # Model instances
        self.models: Dict[str, BaseGECModel] = {}

        # Initialize models
        self._init_models()

        logger.info(f"Ensemble initialized with {len(self.models)} models")

    def _init_models(self) -> None:
        """Initialize all model instances."""

        # Rule-based models (always fast, load immediately)
        rule_based = {
            'taa_marbuta_fixer': TaaMarbutaFixer(),
            'alif_maksura_fixer': AlifMaksuraFixer(),
            'punctuation_fixer': PunctuationFixer(),
            'repeated_word_fixer': RepeatedWordFixer(),
        }

        # Neural models (lazy load by default)
        neural = {
            'hamza_fixer': lambda: HamzaFixer(device=self.config.device),
            'space_fixer': lambda: SpaceFixer(device=self.config.device),
            'deleted_word_fixer': lambda: DeletedWordFixer(device=self.config.device),
            'spelling_fixer': lambda: SpellingFixer(device=self.config.device),
            'general_gec': lambda: GeneralGEC(device=self.config.device),
            # morphology_fixer would use CAMeLBERT - to be implemented
        }

        # Add rule-based models
        for name, model in rule_based.items():
            if self._should_enable(name):
                self.models[name] = model

        # Add neural models
        for name, model_factory in neural.items():
            if self._should_enable(name):
                if self.lazy_load:
                    # Store factory for lazy loading
                    self.models[name] = model_factory
                else:
                    model = model_factory()
                    model.load()
                    self.models[name] = model

    def _should_enable(self, model_name: str) -> bool:
        """Check if a model should be enabled."""
        if self.enabled_models is None:
            return True
        return model_name in self.enabled_models

    def _get_model(self, name: str) -> Optional[BaseGECModel]:
        """Get a model, loading if necessary."""
        if name not in self.models:
            return None

        model = self.models[name]

        # Lazy loading: if it's a factory function, call it
        if callable(model) and not isinstance(model, BaseGECModel):
            logger.info(f"Lazy loading model: {name}")
            model = model()
            model.load()
            self.models[name] = model

        return model

    def correct(
        self,
        text: str,
        strategy: str = "cascading",
        confidence_threshold: float = 0.7,
    ) -> EnsembleResult:
        """
        Correct text using the ensemble.

        Args:
            text: Input text with potential errors
            strategy: 'cascading', 'parallel', or 'specialist'
            confidence_threshold: Minimum confidence to apply correction

        Returns:
            EnsembleResult with corrections
        """
        if strategy == "cascading":
            return self._correct_cascading(text, confidence_threshold)
        elif strategy == "parallel":
            return self._correct_parallel(text, confidence_threshold)
        elif strategy == "specialist":
            return self._correct_specialist(text, confidence_threshold)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _correct_cascading(
        self,
        text: str,
        confidence_threshold: float
    ) -> EnsembleResult:
        """
        Cascading correction strategy.

        1. Apply fast rule-based models first
        2. Then apply neural models in priority order
        3. Each model sees the output of previous models
        """
        current_text = text
        all_corrections = []
        model_contributions = {}

        # Order: rule-based first (fast), then neural by priority
        model_order = [
            # Rule-based (instant)
            'punctuation_fixer',
            'repeated_word_fixer',
            'taa_marbuta_fixer',
            'alif_maksura_fixer',
            # Neural (by priority)
            'hamza_fixer',
            'space_fixer',
            'deleted_word_fixer',
            'spelling_fixer',
            'general_gec',
        ]

        for model_name in model_order:
            model = self._get_model(model_name)
            if model is None:
                continue

            result = model.correct(current_text)

            # Filter by confidence
            confident_corrections = [
                c for c in result.corrections
                if c.confidence >= confidence_threshold
            ]

            if confident_corrections:
                current_text = result.corrected_text
                all_corrections.extend(confident_corrections)
                model_contributions[model_name] = len(confident_corrections)

        # Calculate overall confidence
        if all_corrections:
            overall_confidence = sum(c.confidence for c in all_corrections) / len(all_corrections)
        else:
            overall_confidence = 1.0  # No corrections = confident it's correct

        return EnsembleResult(
            original_text=text,
            corrected_text=current_text,
            corrections=all_corrections,
            model_contributions=model_contributions,
            confidence=overall_confidence
        )

    def _correct_parallel(
        self,
        text: str,
        confidence_threshold: float
    ) -> EnsembleResult:
        """
        Parallel correction strategy.

        1. Run all models on the original text
        2. Collect all corrections
        3. Resolve conflicts by confidence/priority
        4. Apply non-conflicting corrections
        """
        all_results: Dict[str, CorrectionResult] = {}

        # Run all models
        for model_name in self.models:
            model = self._get_model(model_name)
            if model is None:
                continue
            all_results[model_name] = model.correct(text)

        # Collect all corrections with model info
        corrections_with_source = []
        for model_name, result in all_results.items():
            model_config = self.config.models.get(model_name)
            priority = model_config.priority if model_config else 5

            for correction in result.corrections:
                if correction.confidence >= confidence_threshold:
                    corrections_with_source.append((correction, model_name, priority))

        # Sort by priority (highest first), then confidence
        corrections_with_source.sort(key=lambda x: (x[2], x[0].confidence), reverse=True)

        # Apply non-conflicting corrections
        applied_corrections = []
        applied_ranges = []  # (start, end) of applied corrections
        model_contributions = {}

        for correction, model_name, _ in corrections_with_source:
            # Check for overlap with already applied corrections
            overlaps = any(
                not (correction.end_idx <= start or correction.start_idx >= end)
                for start, end in applied_ranges
            )

            if not overlaps:
                applied_corrections.append(correction)
                applied_ranges.append((correction.start_idx, correction.end_idx))
                model_contributions[model_name] = model_contributions.get(model_name, 0) + 1

        # Apply corrections to text (in reverse order to preserve indices)
        corrected_text = text
        for correction in sorted(applied_corrections, key=lambda c: c.start_idx, reverse=True):
            corrected_text = (
                corrected_text[:correction.start_idx] +
                correction.corrected +
                corrected_text[correction.end_idx:]
            )

        # Calculate confidence
        if applied_corrections:
            overall_confidence = sum(c.confidence for c in applied_corrections) / len(applied_corrections)
        else:
            overall_confidence = 1.0

        return EnsembleResult(
            original_text=text,
            corrected_text=corrected_text,
            corrections=applied_corrections,
            model_contributions=model_contributions,
            confidence=overall_confidence
        )

    def _correct_specialist(
        self,
        text: str,
        confidence_threshold: float
    ) -> EnsembleResult:
        """
        Specialist routing strategy.

        1. Analyze text to detect error types
        2. Route to specialist models based on detected types
        3. Combine results
        """
        # First, run general model to get an idea of errors
        general = self._get_model('general_gec')
        if general is None:
            return self._correct_cascading(text, confidence_threshold)

        general_result = general.correct(text)

        # Analyze error types
        error_types = set()
        for correction in general_result.corrections:
            error_types.add(correction.error_type)

        # Route to specialists
        specialist_map = {
            'hamza': 'hamza_fixer',
            'hamza_alif': 'hamza_fixer',
            'taa_marbuta': 'taa_marbuta_fixer',
            'alif_maksura': 'alif_maksura_fixer',
            'merge': 'space_fixer',
            'split': 'space_fixer',
            'repeated_word': 'repeated_word_fixer',
            'spelling': 'spelling_fixer',
        }

        specialists_to_use = set()
        for error_type in error_types:
            if error_type in specialist_map:
                specialists_to_use.add(specialist_map[error_type])

        # If no specialists matched, use cascading
        if not specialists_to_use:
            return self._correct_cascading(text, confidence_threshold)

        # Run specialists and merge
        current_text = text
        all_corrections = []
        model_contributions = {}

        for model_name in specialists_to_use:
            model = self._get_model(model_name)
            if model is None:
                continue

            result = model.correct(current_text)

            confident_corrections = [
                c for c in result.corrections
                if c.confidence >= confidence_threshold
            ]

            if confident_corrections:
                current_text = result.corrected_text
                all_corrections.extend(confident_corrections)
                model_contributions[model_name] = len(confident_corrections)

        if all_corrections:
            overall_confidence = sum(c.confidence for c in all_corrections) / len(all_corrections)
        else:
            overall_confidence = 1.0

        return EnsembleResult(
            original_text=text,
            corrected_text=current_text,
            corrections=all_corrections,
            model_contributions=model_contributions,
            confidence=overall_confidence
        )

    def correct_batch(
        self,
        texts: List[str],
        strategy: str = "cascading",
        confidence_threshold: float = 0.7,
    ) -> List[EnsembleResult]:
        """Correct a batch of texts."""
        return [
            self.correct(text, strategy, confidence_threshold)
            for text in texts
        ]

    def get_model_info(self) -> Dict[str, dict]:
        """Get information about loaded models."""
        info = {}
        for name in self.models:
            model = self._get_model(name)
            if model:
                info[name] = {
                    'name': model.name,
                    'error_types': model.error_types,
                    'is_loaded': model.is_loaded,
                    'type': type(model).__name__,
                }
        return info

    def enable_model(self, name: str) -> bool:
        """Enable a specific model."""
        if name in self.models:
            return True

        # Try to initialize it
        model_config = self.config.models.get(name)
        if model_config:
            model_config.enabled = True
            self._init_models()
            return name in self.models

        return False

    def disable_model(self, name: str) -> bool:
        """Disable a specific model."""
        if name in self.models:
            del self.models[name]
            return True
        return False


# Convenience function
def correct(text: str, **kwargs) -> EnsembleResult:
    """
    Correct text using the default ensemble.

    This creates a new ensemble each time, so for batch processing,
    create an ensemble instance and reuse it.
    """
    ensemble = NahawiEnsemble(**kwargs)
    return ensemble.correct(text)
