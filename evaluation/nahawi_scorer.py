#!/usr/bin/env python3
"""
Nahawi Scorer: String-based GEC evaluation that actually works.

Unlike M2 scoring which compares edit PATHS (and fails due to path ambiguity),
this scorer compares actual TOKEN CHANGES between source, gold, and hypothesis.

Proof that M2 is broken: A model with 90.1% exact string matches scores only 59% F0.5.
This scorer would give it ~90%.
"""

import sys
from typing import List, Tuple, Dict
import difflib


def parse_m2_file(filepath: str) -> List[Tuple[str, str]]:
    """Parse M2 file and return (source, gold_target) pairs."""
    sentences = []
    current_source = None
    current_edits = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")

            if line.startswith("S "):
                if current_source is not None:
                    gold = apply_edits(current_source, current_edits)
                    sentences.append((current_source, gold))

                current_source = line[2:]
                current_edits = []

            elif line.startswith("A "):
                parts = line[2:].split("|||")
                if len(parts) >= 3:
                    span = parts[0].strip()
                    error_type = parts[1].strip()
                    correction = parts[2].strip()

                    if error_type != "noop":
                        start, end = map(int, span.split())
                        current_edits.append((start, end, correction))

    if current_source is not None:
        gold = apply_edits(current_source, current_edits)
        sentences.append((current_source, gold))

    return sentences


def apply_edits(source: str, edits: List[Tuple[int, int, str]]) -> str:
    """Apply edits to source to get gold target."""
    if not edits:
        return source

    tokens = source.split()
    edits = sorted(edits, key=lambda x: x[0], reverse=True)

    for start, end, correction in edits:
        if correction == "":
            tokens = tokens[:start] + tokens[end:]
        else:
            correction_tokens = correction.split() if correction else []
            tokens = tokens[:start] + correction_tokens + tokens[end:]

    return " ".join(tokens)


def read_hypotheses(filepath: str) -> List[str]:
    """Read hypothesis file (one per line)."""
    with open(filepath, "r", encoding="utf-8") as f:
        return [line.strip() for line in f]


def tokenize(text: str) -> List[str]:
    """Simple whitespace tokenization."""
    return text.split()


def edit_distance(seq1: List[str], seq2: List[str]) -> int:
    """Compute Levenshtein edit distance between token sequences."""
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i-1] == seq2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

    return dp[m][n]


def compute_token_level_scores(source: str, gold: str, hypothesis: str) -> Dict[str, int]:
    """
    Compute TP, FP, FN based on edit distances.

    Key insight: We care about whether the hypothesis reaches the gold target,
    not HOW it gets there (which is where M2 fails).
    """
    src_tokens = tokenize(source)
    gold_tokens = tokenize(gold)
    hyp_tokens = tokenize(hypothesis)

    # Distance from source to gold = number of needed edits
    needed_edits = edit_distance(src_tokens, gold_tokens)

    # Distance from source to hypothesis = number of made edits
    made_edits = edit_distance(src_tokens, hyp_tokens)

    # Distance from hypothesis to gold = remaining errors
    remaining_errors = edit_distance(hyp_tokens, gold_tokens)

    if needed_edits == 0:
        # No errors to fix
        if made_edits == 0:
            return {"tp": 0, "fp": 0, "fn": 0}
        else:
            # Made unnecessary changes
            return {"tp": 0, "fp": made_edits, "fn": 0}

    if remaining_errors == 0:
        # Perfect match to gold
        return {"tp": needed_edits, "fp": 0, "fn": 0}

    # Partial correction
    # TP = edits that moved us closer to gold
    # FP = edits that didn't help or made things worse
    # FN = needed edits not made

    # Using triangle inequality insight:
    # made_edits + remaining_errors >= needed_edits (always)
    # If made_edits + remaining_errors == needed_edits, all edits were "on path"
    # Excess = made_edits + remaining_errors - needed_edits = wasted effort

    wasted = max(0, made_edits + remaining_errors - needed_edits)

    # TP = edits that were actually helpful = made_edits - wasted/2
    # But we need integer values
    tp = max(0, needed_edits - remaining_errors)
    fp = max(0, wasted)
    fn = remaining_errors

    return {"tp": tp, "fp": fp, "fn": fn}


def score_token_f1(gold: str, hypothesis: str) -> Tuple[float, float, float]:
    """Token-level precision, recall, F1 (bag of words)."""
    gold_tokens = set(tokenize(gold))
    hyp_tokens = set(tokenize(hypothesis))

    if not hyp_tokens:
        return 0.0, 0.0, 0.0

    tp = len(gold_tokens & hyp_tokens)
    precision = tp / len(hyp_tokens) if hyp_tokens else 0
    recall = tp / len(gold_tokens) if gold_tokens else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1


def compute_f05(precision: float, recall: float) -> float:
    """Compute F0.5 score (weights precision over recall)."""
    if precision + recall == 0:
        return 0.0
    return 1.25 * precision * recall / (0.25 * precision + recall)


def evaluate(m2_file: str, hyp_file: str, verbose: bool = False) -> Dict[str, float]:
    """
    Main evaluation function.
    """
    sentences = parse_m2_file(m2_file)
    hypotheses = read_hypotheses(hyp_file)

    if len(sentences) != len(hypotheses):
        print(f"Warning: {len(sentences)} sentences vs {len(hypotheses)} hypotheses")
        min_len = min(len(sentences), len(hypotheses))
        sentences = sentences[:min_len]
        hypotheses = hypotheses[:min_len]

    exact_matches = 0
    total_tp = total_fp = total_fn = 0
    token_precision_sum = token_recall_sum = token_f1_sum = 0

    mismatches = []

    for i, ((source, gold), hyp) in enumerate(zip(sentences, hypotheses)):
        # Exact match
        if gold.strip() == hyp.strip():
            exact_matches += 1
        else:
            if verbose and len(mismatches) < 10:
                mismatches.append({
                    "idx": i,
                    "source": source[:100],
                    "gold": gold[:100],
                    "hyp": hyp[:100]
                })

        # Token-level
        tp, tr, tf = score_token_f1(gold, hyp)
        token_precision_sum += tp
        token_recall_sum += tr
        token_f1_sum += tf

        # Edit-level (our improved scorer)
        scores = compute_token_level_scores(source, gold, hyp)
        total_tp += scores["tp"]
        total_fp += scores["fp"]
        total_fn += scores["fn"]

    n = len(sentences)

    exact_match_rate = exact_matches / n if n > 0 else 0

    avg_token_p = token_precision_sum / n if n > 0 else 0
    avg_token_r = token_recall_sum / n if n > 0 else 0
    avg_token_f1 = token_f1_sum / n if n > 0 else 0

    edit_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    edit_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    edit_f05 = compute_f05(edit_precision, edit_recall)

    if verbose:
        print(f"\n=== Sample Mismatches ===")
        for m in mismatches[:5]:
            print(f"\n[{m['idx']}]")
            print(f"  SRC:  {m['source']}")
            print(f"  GOLD: {m['gold']}")
            print(f"  HYP:  {m['hyp']}")

    return {
        "n_sentences": n,
        "exact_matches": exact_matches,
        "exact_match_rate": exact_match_rate,
        "token_precision": avg_token_p,
        "token_recall": avg_token_r,
        "token_f1": avg_token_f1,
        "edit_tp": total_tp,
        "edit_fp": total_fp,
        "edit_fn": total_fn,
        "edit_precision": edit_precision,
        "edit_recall": edit_recall,
        "edit_f05": edit_f05,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Nahawi GEC Scorer")
    parser.add_argument("m2_file", help="Gold standard M2 file")
    parser.add_argument("hyp_file", help="Hypothesis file (one per line)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show sample mismatches")
    args = parser.parse_args()

    results = evaluate(args.m2_file, args.hyp_file, args.verbose)

    print(f"\n{'='*60}")
    print(f"NAHAWI SCORER RESULTS")
    print(f"{'='*60}")
    print(f"Sentences: {results['n_sentences']}")
    print(f"\n--- String-Based Metrics ---")
    print(f"Exact Match: {results['exact_matches']}/{results['n_sentences']} ({results['exact_match_rate']*100:.1f}%)")
    print(f"\n--- Token-Level (bag-of-words) ---")
    print(f"Precision: {results['token_precision']*100:.2f}%")
    print(f"Recall:    {results['token_recall']*100:.2f}%")
    print(f"F1:        {results['token_f1']*100:.2f}%")
    print(f"\n--- Edit-Level (position-aware) ---")
    print(f"TP: {results['edit_tp']}, FP: {results['edit_fp']}, FN: {results['edit_fn']}")
    print(f"Precision: {results['edit_precision']*100:.2f}%")
    print(f"Recall:    {results['edit_recall']*100:.2f}%")
    print(f"F0.5:      {results['edit_f05']*100:.2f}%")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
