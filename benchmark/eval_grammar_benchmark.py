#!/usr/bin/env python3
"""
FASIH v6 Grammar Benchmark Evaluation

Evaluates Nahawi model on grammar-specific test cases.
Reports per-category accuracy and overall grammar F0.5.
"""

import json
import sys
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_benchmark(benchmark_path: Path) -> List[dict]:
    """Load benchmark JSON file."""
    with open(benchmark_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def compute_word_level_metrics(source: str, target: str, prediction: str) -> Dict:
    """
    Compute word-level precision, recall, F0.5.

    Returns dict with:
    - tp: true positives (correctly identified corrections)
    - fp: false positives (incorrect changes)
    - fn: false negatives (missed corrections)
    """
    source_words = source.split()
    target_words = target.split()
    pred_words = prediction.split()

    # Find corrections needed (source -> target)
    needed_corrections = set()
    for i, (s, t) in enumerate(zip(source_words, target_words)):
        if s != t:
            needed_corrections.add((i, s, t))

    # Find corrections made (source -> prediction)
    made_corrections = set()
    for i, (s, p) in enumerate(zip(source_words, pred_words)):
        if s != p:
            made_corrections.add((i, s, p))

    # Calculate TP, FP, FN
    tp = 0
    for i, s, t in needed_corrections:
        # Check if prediction made this correction
        for j, ps, pp in made_corrections:
            if i == j and pp == t:
                tp += 1
                break

    fp = len(made_corrections) - tp
    fn = len(needed_corrections) - tp

    return {'tp': tp, 'fp': fp, 'fn': fn}


def evaluate_grammar_benchmark(
    model_correct_fn,
    benchmark_path: Path,
    verbose: bool = True
) -> Dict:
    """
    Evaluate model on grammar benchmark.

    Args:
        model_correct_fn: Function that takes source text and returns corrected text
        benchmark_path: Path to benchmark JSON
        verbose: Print progress

    Returns:
        Dict with per-category and overall metrics
    """
    samples = load_benchmark(benchmark_path)

    if verbose:
        print(f"\nEvaluating on {len(samples)} samples...")
        print("=" * 60)

    # Track metrics per category
    category_metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'total': 0, 'exact_match': 0})

    # Track individual results for analysis
    results = []

    start_time = time.time()

    for i, sample in enumerate(samples):
        source = sample['source']
        target = sample['target']
        error_type = sample['error_type']

        # Get model prediction
        try:
            prediction = model_correct_fn(source)
        except Exception as e:
            prediction = source  # No correction on error
            if verbose:
                print(f"  Error on sample {i}: {e}")

        # Compute metrics
        metrics = compute_word_level_metrics(source, target, prediction)

        category_metrics[error_type]['tp'] += metrics['tp']
        category_metrics[error_type]['fp'] += metrics['fp']
        category_metrics[error_type]['fn'] += metrics['fn']
        category_metrics[error_type]['total'] += 1

        # Exact match
        if prediction.strip() == target.strip():
            category_metrics[error_type]['exact_match'] += 1

        results.append({
            'source': source,
            'target': target,
            'prediction': prediction,
            'error_type': error_type,
            'correct': prediction.strip() == target.strip(),
            'metrics': metrics
        })

        if verbose and (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(samples)}...")

    elapsed = time.time() - start_time

    # Calculate F0.5 per category
    category_results = {}
    total_tp, total_fp, total_fn = 0, 0, 0

    for category, m in category_metrics.items():
        tp, fp, fn = m['tp'], m['fp'], m['fn']
        total_tp += tp
        total_fp += fp
        total_fn += fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        beta = 0.5
        f05 = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall) if (precision + recall) > 0 else 0

        exact_match_rate = m['exact_match'] / m['total'] if m['total'] > 0 else 0

        category_results[category] = {
            'precision': precision * 100,
            'recall': recall * 100,
            'f05': f05 * 100,
            'exact_match': exact_match_rate * 100,
            'total': m['total'],
            'tp': tp,
            'fp': fp,
            'fn': fn
        }

    # Overall metrics
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0

    beta = 0.5
    overall_f05 = (1 + beta**2) * (overall_precision * overall_recall) / ((beta**2 * overall_precision) + overall_recall) if (overall_precision + overall_recall) > 0 else 0

    overall = {
        'precision': overall_precision * 100,
        'recall': overall_recall * 100,
        'f05': overall_f05 * 100,
        'total_samples': len(samples),
        'elapsed_seconds': elapsed
    }

    if verbose:
        print("\n" + "=" * 60)
        print("GRAMMAR BENCHMARK RESULTS")
        print("=" * 60)

        print(f"\n{'Category':<20} {'F0.5':>8} {'Prec':>8} {'Recall':>8} {'Exact':>8} {'N':>6}")
        print("-" * 60)

        for category in sorted(category_results.keys()):
            r = category_results[category]
            print(f"{category:<20} {r['f05']:>7.1f}% {r['precision']:>7.1f}% {r['recall']:>7.1f}% {r['exact_match']:>7.1f}% {r['total']:>6}")

        print("-" * 60)
        print(f"{'OVERALL':<20} {overall['f05']:>7.1f}% {overall['precision']:>7.1f}% {overall['recall']:>7.1f}%")
        print(f"\nTime: {elapsed:.1f}s ({len(samples)/elapsed:.1f} samples/sec)")

    return {
        'overall': overall,
        'categories': category_results,
        'results': results
    }


def run_with_nahawi_model():
    """Run evaluation with the Nahawi model."""

    # Try to load model
    try:
        from nahawi import NahawiModel

        print("Loading Nahawi model...")
        model = NahawiModel()
        model.load()

        def correct_fn(text: str) -> str:
            corrected, _ = model.correct(text)
            return corrected

    except Exception as e:
        print(f"Could not load Nahawi model: {e}")
        print("Using identity function (baseline)...")

        def correct_fn(text: str) -> str:
            return text  # No correction

    # Find benchmark
    benchmark_path = Path(__file__).parent / "fasih_v6_grammar" / "fasih_v6_grammar_test.json"

    if not benchmark_path.exists():
        print(f"Benchmark not found: {benchmark_path}")
        print("Run build_fasih_v6_grammar.py first.")
        return

    # Run evaluation
    results = evaluate_grammar_benchmark(correct_fn, benchmark_path)

    # Save results
    output_path = Path(__file__).parent / "fasih_v6_grammar" / "evaluation_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        # Don't save individual results (too large), just summary
        summary = {
            'overall': results['overall'],
            'categories': results['categories']
        }
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {output_path}")

    # Print interpretation
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)

    overall_f05 = results['overall']['f05']

    if overall_f05 < 10:
        print("Grammar correction: CRITICAL - Model has near-zero grammar capability")
        print("Action: Grammar-focused LoRA training is essential")
    elif overall_f05 < 30:
        print("Grammar correction: POOR - Model occasionally corrects grammar")
        print("Action: Significant grammar training data needed")
    elif overall_f05 < 50:
        print("Grammar correction: FAIR - Model has some grammar awareness")
        print("Action: More grammar training will help")
    elif overall_f05 < 70:
        print("Grammar correction: GOOD - Model handles many grammar errors")
        print("Action: Fine-tuning on specific weak categories")
    else:
        print("Grammar correction: EXCELLENT - Strong grammar capabilities")

    # Identify weakest categories
    print("\nWeakest categories (priority for training):")
    sorted_cats = sorted(results['categories'].items(), key=lambda x: x[1]['f05'])
    for cat, metrics in sorted_cats[:3]:
        print(f"  - {cat}: {metrics['f05']:.1f}% F0.5")


if __name__ == "__main__":
    run_with_nahawi_model()
