#!/usr/bin/env python3
"""
Evaluate the Nahawi ensemble on QALB data.

Usage:
    python evaluate.py                    # Evaluate full ensemble
    python evaluate.py --models general_gec  # Evaluate single model
    python evaluate.py --strategy parallel   # Use parallel strategy
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Dict
import json
from collections import defaultdict

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nahawi_ensemble.orchestrator import NahawiEnsemble, EnsembleResult
from nahawi_ensemble.config import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def load_pairs(path: Path, max_pairs: int = None) -> List[Tuple[str, str]]:
    """Load error-correct pairs from TSV."""
    pairs = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_pairs and i >= max_pairs:
                break
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                pairs.append((parts[0], parts[1]))
    return pairs


def calculate_metrics(
    predictions: List[str],
    sources: List[str],
    targets: List[str]
) -> Dict[str, float]:
    """
    Calculate GEC metrics.

    Returns:
        Dict with precision, recall, F0.5, exact_match
    """
    total_tp, total_fp, total_fn = 0, 0, 0
    exact_match = 0

    for pred, src, tgt in zip(predictions, sources, targets):
        pred = pred.strip()
        tgt = tgt.strip()

        if pred == tgt:
            exact_match += 1

        # Word-level edits
        src_words = src.split()
        pred_words = pred.split()
        tgt_words = tgt.split()

        # Calculate edits made by prediction
        pred_edits = set()
        for i, (s, p) in enumerate(zip(src_words, pred_words)):
            if s != p:
                pred_edits.add((i, s, p))
        for i in range(len(src_words), len(pred_words)):
            pred_edits.add((i, '', pred_words[i]))
        for i in range(len(pred_words), len(src_words)):
            pred_edits.add((i, src_words[i], ''))

        # Calculate gold edits
        gold_edits = set()
        for i, (s, t) in enumerate(zip(src_words, tgt_words)):
            if s != t:
                gold_edits.add((i, s, t))
        for i in range(len(src_words), len(tgt_words)):
            gold_edits.add((i, '', tgt_words[i]))
        for i in range(len(tgt_words), len(src_words)):
            gold_edits.add((i, src_words[i], ''))

        total_tp += len(pred_edits & gold_edits)
        total_fp += len(pred_edits - gold_edits)
        total_fn += len(gold_edits - pred_edits)

    # Calculate metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f05 = (1.25 * precision * recall) / (0.25 * precision + recall) if (precision + recall) > 0 else 0

    return {
        'precision': precision,
        'recall': recall,
        'f05': f05,
        'exact_match': exact_match / len(predictions) if predictions else 0,
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn,
        'total': len(predictions),
    }


def evaluate_ensemble(
    ensemble: NahawiEnsemble,
    pairs: List[Tuple[str, str]],
    strategy: str = 'cascading',
    confidence_threshold: float = 0.7,
) -> Dict:
    """Evaluate the ensemble on pairs."""
    sources = [p[0] for p in pairs]
    targets = [p[1] for p in pairs]

    logger.info(f"Evaluating {len(pairs)} pairs with strategy '{strategy}'...")

    # Get predictions
    predictions = []
    model_stats = defaultdict(int)

    for i, (src, tgt) in enumerate(pairs):
        result = ensemble.correct(src, strategy=strategy, confidence_threshold=confidence_threshold)
        predictions.append(result.corrected_text)

        for model, count in result.model_contributions.items():
            model_stats[model] += count

        if (i + 1) % 100 == 0:
            logger.info(f"  Processed {i + 1}/{len(pairs)}")

    # Calculate metrics
    metrics = calculate_metrics(predictions, sources, targets)

    return {
        'metrics': metrics,
        'model_contributions': dict(model_stats),
        'strategy': strategy,
        'confidence_threshold': confidence_threshold,
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate Nahawi ensemble')
    parser.add_argument('--data', type=Path, default=config.data_dir / 'dev.tsv', help='Evaluation data')
    parser.add_argument('--max-samples', type=int, default=None, help='Max samples to evaluate')
    parser.add_argument('--strategy', type=str, default='cascading', choices=['cascading', 'parallel', 'specialist'])
    parser.add_argument('--confidence', type=float, default=0.7, help='Confidence threshold')
    parser.add_argument('--models', type=str, nargs='+', default=None, help='Specific models to enable')
    parser.add_argument('--output', type=Path, default=None, help='Save results to JSON')
    args = parser.parse_args()

    # Load data
    if not args.data.exists():
        logger.error(f"Data file not found: {args.data}")
        return

    pairs = load_pairs(args.data, args.max_samples)
    logger.info(f"Loaded {len(pairs)} pairs from {args.data}")

    # Create ensemble
    ensemble = NahawiEnsemble(
        enabled_models=args.models,
        lazy_load=True
    )

    # Show enabled models
    model_info = ensemble.get_model_info()
    logger.info(f"Enabled models: {list(model_info.keys())}")

    # Evaluate
    results = evaluate_ensemble(
        ensemble,
        pairs,
        strategy=args.strategy,
        confidence_threshold=args.confidence
    )

    # Print results
    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)

    metrics = results['metrics']
    logger.info(f"F0.5:        {metrics['f05']:.4f} ({metrics['f05']*100:.2f}%)")
    logger.info(f"Precision:   {metrics['precision']:.4f}")
    logger.info(f"Recall:      {metrics['recall']:.4f}")
    logger.info(f"Exact Match: {metrics['exact_match']:.4f}")
    logger.info(f"TP/FP/FN:    {metrics['tp']}/{metrics['fp']}/{metrics['fn']}")

    logger.info("\nModel contributions:")
    for model, count in sorted(results['model_contributions'].items(), key=lambda x: -x[1]):
        logger.info(f"  {model}: {count} corrections")

    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
