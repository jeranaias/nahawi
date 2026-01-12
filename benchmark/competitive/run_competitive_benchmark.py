#!/usr/bin/env python3
"""
Run Nahawi on the Competitive Test Set V4 (All 13 FASIH Error Types).

Produces:
- Correction rate by error type (13 categories)
- Per-sentence results
- Comparison data for Word/Google testing
"""

import sys
import json
from pathlib import Path
from typing import List, Dict, Tuple

# Add nahawi_ensemble to path (two levels up from competitive/)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def normalize_text(text: str) -> str:
    """Normalize Arabic text for comparison (strip diacritics)."""
    DIACRITICS = [
        '\u064B',  # Fathatan
        '\u064C',  # Dammatan
        '\u064D',  # Kasratan
        '\u064E',  # Fatha
        '\u064F',  # Damma
        '\u0650',  # Kasra
        '\u0651',  # Shadda
        '\u0652',  # Sukun
        '\u0670',  # Superscript alef
    ]
    for d in DIACRITICS:
        text = text.replace(d, '')
    return text.strip()


def compute_correction_stats(source: str, hyp: str, target: str) -> Dict:
    """Compute word-level correction statistics."""
    src_words = source.split()
    hyp_words = normalize_text(hyp).split()
    tgt_words = target.split()

    # Pad to same length
    max_len = max(len(src_words), len(hyp_words), len(tgt_words))
    src_words += [''] * (max_len - len(src_words))
    hyp_words += [''] * (max_len - len(hyp_words))
    tgt_words += [''] * (max_len - len(tgt_words))

    tp = fp = fn = 0

    for src, hyp, tgt in zip(src_words, hyp_words, tgt_words):
        has_error = (src != tgt)
        made_change = (src != hyp)

        if has_error:
            if made_change and hyp == tgt:
                tp += 1  # Correct fix
            elif made_change:
                fp += 1  # Wrong fix
                fn += 1  # Still missed
            else:
                fn += 1  # Missed
        elif made_change:
            fp += 1  # False positive

    return {'tp': tp, 'fp': fp, 'fn': fn}


def load_test_set(json_path: str) -> List[Dict]:
    """Load competitive test set from JSON."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['test_cases']


def run_benchmark_with_ensemble():
    """Run benchmark using NahawiEnsemble."""
    try:
        from nahawi_ensemble.orchestrator import NahawiEnsemble

        print("Loading Nahawi Ensemble...")
        ensemble = NahawiEnsemble(lazy_load=True)

        # Load test set
        test_path = Path(__file__).parent / "competitive_test_set_v4.json"
        test_cases = load_test_set(str(test_path))

        print(f"Running on {len(test_cases)} sentences...")

        results = []
        total_stats = {'tp': 0, 'fp': 0, 'fn': 0}
        error_type_stats = {}

        for i, case in enumerate(test_cases, 1):
            source = case['source']
            target = case['target']
            errors = case.get('errors', [])

            # Run correction
            result = ensemble.correct(source)
            hyp = result.corrected_text

            # Compute stats
            stats = compute_correction_stats(source, hyp, target)

            total_stats['tp'] += stats['tp']
            total_stats['fp'] += stats['fp']
            total_stats['fn'] += stats['fn']

            # Track by error type
            for err_type in set(errors):
                if err_type not in error_type_stats:
                    error_type_stats[err_type] = {'tp': 0, 'fp': 0, 'fn': 0}
                # Approximation: distribute stats by error type count
                err_count = errors.count(err_type)
                err_weight = err_count / len(errors) if errors else 0
                error_type_stats[err_type]['tp'] += int(stats['tp'] * err_weight)
                error_type_stats[err_type]['fn'] += int(stats['fn'] * err_weight)

            results.append({
                'id': case['id'],
                'source': source,
                'hypothesis': hyp,
                'target': target,
                'stats': stats,
                'exact_match': normalize_text(hyp) == target
            })

            if i % 10 == 0:
                print(f"  Processed {i}/{len(test_cases)}...")

        return results, total_stats, error_type_stats

    except ImportError as e:
        print(f"Could not import NahawiEnsemble: {e}")
        print("Using rule-based only mode...")
        return run_benchmark_rule_based_only()


def run_benchmark_rule_based_only():
    """Run benchmark using only rule-based fixers (no neural models required)."""
    try:
        from nahawi_ensemble.models.rule_based import (
            TaaMarbutaFixer, AlifMaksuraFixer, HamzaFixerRuleBased
        )

        print("Loading rule-based models only...")
        taa_fixer = TaaMarbutaFixer()
        alif_fixer = AlifMaksuraFixer()
        hamza_fixer = HamzaFixerRuleBased()

        # Load test set
        test_path = Path(__file__).parent / "competitive_test_set_v4.json"
        test_cases = load_test_set(str(test_path))

        print(f"Running on {len(test_cases)} sentences...")

        results = []
        total_stats = {'tp': 0, 'fp': 0, 'fn': 0}

        for i, case in enumerate(test_cases, 1):
            source = case['source']
            target = case['target']

            # Apply rule-based fixers in cascade
            text = source
            text = taa_fixer.correct(text).corrected_text
            text = alif_fixer.correct(text).corrected_text
            text = hamza_fixer.correct(text).corrected_text
            hyp = text

            # Compute stats
            stats = compute_correction_stats(source, hyp, target)

            total_stats['tp'] += stats['tp']
            total_stats['fp'] += stats['fp']
            total_stats['fn'] += stats['fn']

            results.append({
                'id': case['id'],
                'source': source,
                'hypothesis': hyp,
                'target': target,
                'stats': stats,
                'exact_match': normalize_text(hyp) == target
            })

            if i % 10 == 0:
                print(f"  Processed {i}/{len(test_cases)}...")

        return results, total_stats, {}

    except ImportError as e:
        print(f"Could not import rule-based models: {e}")
        return None, None, None


def main():
    print("=" * 60)
    print("Nahawi Competitive Benchmark V4 (All 13 FASIH Error Types)")
    print("=" * 60)

    results, total_stats, error_type_stats = run_benchmark_with_ensemble()

    if results is None:
        print("ERROR: Could not run benchmark. Check model availability.")
        return

    # Compute metrics
    tp = total_stats['tp']
    fp = total_stats['fp']
    fn = total_stats['fn']

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f05 = (1 + 0.5**2) * (precision * recall) / (0.5**2 * precision + recall) if (precision + recall) > 0 else 0

    exact_matches = sum(1 for r in results if r['exact_match'])
    correction_rate = tp / (tp + fn) if (tp + fn) > 0 else 0

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Total sentences: {len(results)}")
    print(f"Exact matches: {exact_matches}/{len(results)} ({100*exact_matches/len(results):.1f}%)")
    print(f"\nWord-level metrics:")
    print(f"  TP: {tp}, FP: {fp}, FN: {fn}")
    print(f"  Precision: {100*precision:.1f}%")
    print(f"  Recall: {100*recall:.1f}%")
    print(f"  F0.5: {100*f05:.1f}%")
    print(f"  Correction Rate: {100*correction_rate:.1f}%")

    if error_type_stats:
        print("\nBy error type:")
        for err_type, stats in sorted(error_type_stats.items()):
            err_tp = stats['tp']
            err_fn = stats['fn']
            err_rate = err_tp / (err_tp + err_fn) if (err_tp + err_fn) > 0 else 0
            print(f"  {err_type}: {100*err_rate:.1f}% ({err_tp}/{err_tp + err_fn})")

    # Save results
    output_path = Path(__file__).parent / "nahawi_competitive_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': {
                'total_sentences': len(results),
                'exact_matches': exact_matches,
                'precision': precision,
                'recall': recall,
                'f05': f05,
                'correction_rate': correction_rate
            },
            'results': results
        }, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
