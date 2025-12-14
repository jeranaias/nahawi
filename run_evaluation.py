#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run full evaluation of the Nahawi ensemble.

This script:
1. Checks which models are available (trained)
2. Runs evaluation on QALB dev set
3. Reports F0.5, precision, recall metrics
4. Shows per-model contribution
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import List, Tuple, Dict

if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path('C:/nahawi')))

# Paths
NAHAWI_DIR = Path('C:/nahawi')
CHECKPOINT_DIR = NAHAWI_DIR / 'nahawi_ensemble' / 'checkpoints'
QALB_DIR = NAHAWI_DIR / 'qalb_real_data'


def check_available_models():
    """Check which models are available."""
    available = {
        'rule_based': [],
        'neural': [],
    }

    # Rule-based are always available
    available['rule_based'] = [
        'taa_marbuta_fixer',
        'alif_maksura_fixer',
        'punctuation_fixer',
        'repeated_word_fixer',
    ]

    # Check neural models
    neural_models = [
        'general_gec',
        'hamza_fixer',
        'space_fixer',
        'spelling_fixer',
        'deleted_word_fixer',
        'morphology_fixer',
    ]

    for model in neural_models:
        model_dir = CHECKPOINT_DIR / model / 'final'
        if model_dir.exists():
            # Check for model files
            if (model_dir / 'model.safetensors').exists() or (model_dir / 'pytorch_model.bin').exists():
                available['neural'].append(model)

    return available


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


def calculate_metrics(predictions: List[str], sources: List[str], targets: List[str]) -> Dict:
    """Calculate GEC metrics."""
    total_tp, total_fp, total_fn = 0, 0, 0
    exact_match = 0

    for pred, src, tgt in zip(predictions, sources, targets):
        pred = pred.strip()
        tgt = tgt.strip()

        if pred == tgt:
            exact_match += 1

        # Word-level comparison
        src_words = src.split()
        pred_words = pred.split()
        tgt_words = tgt.split()

        # Simple word-level edit comparison
        pred_edits = set()
        for i, (s, p) in enumerate(zip(src_words, pred_words)):
            if s != p:
                pred_edits.add((i, s, p))

        gold_edits = set()
        for i, (s, t) in enumerate(zip(src_words, tgt_words)):
            if s != t:
                gold_edits.add((i, s, t))

        total_tp += len(pred_edits & gold_edits)
        total_fp += len(pred_edits - gold_edits)
        total_fn += len(gold_edits - pred_edits)

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


def run_evaluation(max_samples: int = 500):
    """Run full evaluation."""
    print("=" * 60)
    print("NAHAWI ENSEMBLE EVALUATION")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Check available models
    available = check_available_models()
    print(f"\nAvailable models:")
    print(f"  Rule-based: {available['rule_based']}")
    print(f"  Neural:     {available['neural']}")

    # Load evaluation data
    dev_file = QALB_DIR / 'dev.tsv'
    if not dev_file.exists():
        print(f"\nError: Dev file not found: {dev_file}")
        return

    pairs = load_pairs(dev_file, max_samples)
    print(f"\nEvaluation data: {len(pairs)} pairs")

    # Initialize ensemble
    from nahawi_ensemble.orchestrator import NahawiEnsemble

    all_models = available['rule_based'] + available['neural']
    if not all_models:
        print("\nNo models available!")
        return

    ensemble = NahawiEnsemble(
        enabled_models=all_models,
        lazy_load=True
    )

    # Run evaluation
    print(f"\nRunning evaluation...")
    predictions = []
    sources = []
    targets = []
    model_stats = defaultdict(int)

    for i, (src, tgt) in enumerate(pairs):
        sources.append(src)
        targets.append(tgt)

        result = ensemble.correct(src, strategy="cascading")
        predictions.append(result.corrected_text)

        for model, count in result.model_contributions.items():
            model_stats[model] += count

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(pairs)}")

    # Calculate metrics
    metrics = calculate_metrics(predictions, sources, targets)

    # Report
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nF0.5:        {metrics['f05']:.4f} ({metrics['f05']*100:.2f}%)")
    print(f"Precision:   {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"Recall:      {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"Exact Match: {metrics['exact_match']:.4f} ({metrics['exact_match']*100:.2f}%)")
    print(f"\nTP: {metrics['tp']}, FP: {metrics['fp']}, FN: {metrics['fn']}")

    print("\nModel Contributions:")
    for model, count in sorted(model_stats.items(), key=lambda x: -x[1]):
        print(f"  {model}: {count} corrections")

    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'samples': len(pairs),
        'models': all_models,
        'metrics': metrics,
        'model_contributions': dict(model_stats),
    }

    results_file = NAHAWI_DIR / 'evaluation_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    print("=" * 60)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-samples', type=int, default=500)
    args = parser.parse_args()

    run_evaluation(args.max_samples)
