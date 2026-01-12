#!/usr/bin/env python3
"""
Extract common punct patterns from QALB gold.
Find mechanical patterns like "؛ وذلك" that can be applied as rules.
"""

import json
from collections import Counter
import re

PUNCT_SET = set('،.؟!؛:,;?')

def is_punct(tok):
    return all(c in PUNCT_SET for c in tok) and len(tok) > 0


def extract_patterns(text, window=2):
    """Extract (before, punct, after) patterns."""
    tokens = text.split()
    patterns = []

    for i, tok in enumerate(tokens):
        if is_punct(tok):
            # Get context before
            before = []
            for j in range(max(0, i-window), i):
                if not is_punct(tokens[j]):
                    before.append(tokens[j])

            # Get context after
            after = []
            for j in range(i+1, min(len(tokens), i+1+window)):
                if not is_punct(tokens[j]):
                    after.append(tokens[j])

            # Create pattern tuple
            before_str = ' '.join(before[-window:]) if before else ''
            after_str = ' '.join(after[:window]) if after else ''

            patterns.append({
                'punct': tok,
                'before': before_str,
                'after': after_str,
                'full': f"{before_str} [{tok}] {after_str}".strip()
            })

    return patterns


def main():
    print("=" * 70)
    print("EXTRACT PUNCT PATTERNS FROM QALB GOLD")
    print("=" * 70)

    # Load QALB train and dev
    files = [
        '/home/ubuntu/nahawi/data/qalb_real_train.json',
        '/home/ubuntu/nahawi/data/qalb_real_dev.json'
    ]

    all_patterns = []
    total_punct = Counter()

    for filepath in files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"Loaded {len(data)} examples from {filepath}")

            for item in data:
                # Use target (gold) text
                text = item.get('target', item.get('tgt', ''))
                patterns = extract_patterns(text)
                all_patterns.extend(patterns)

                # Count punct types
                for tok in text.split():
                    if is_punct(tok):
                        total_punct[tok] += 1

        except Exception as e:
            print(f"  Error loading {filepath}: {e}")

    print(f"\nTotal patterns extracted: {len(all_patterns)}")

    # Count punct types
    print("\n" + "=" * 50)
    print("PUNCT TYPE DISTRIBUTION")
    print("=" * 50)
    for punct, count in total_punct.most_common():
        print(f"  {punct}: {count} ({100*count/sum(total_punct.values()):.1f}%)")

    # Analyze before patterns (what comes before punct)
    print("\n" + "=" * 50)
    print("TOP 30 PATTERNS: WORD BEFORE PUNCT")
    print("=" * 50)
    before_patterns = Counter()
    for p in all_patterns:
        # Last word before punct
        if p['before']:
            last_word = p['before'].split()[-1] if p['before'] else ''
            if last_word:
                key = f"{last_word} → {p['punct']}"
                before_patterns[key] += 1

    for pattern, count in before_patterns.most_common(30):
        print(f"  {pattern}: {count}")

    # Analyze after patterns (what comes after punct)
    print("\n" + "=" * 50)
    print("TOP 30 PATTERNS: PUNCT BEFORE WORD")
    print("=" * 50)
    after_patterns = Counter()
    for p in all_patterns:
        # First word after punct
        if p['after']:
            first_word = p['after'].split()[0] if p['after'] else ''
            if first_word:
                key = f"{p['punct']} → {first_word}"
                after_patterns[key] += 1

    for pattern, count in after_patterns.most_common(30):
        print(f"  {pattern}: {count}")

    # Full context patterns (before + punct + after)
    print("\n" + "=" * 50)
    print("TOP 30 FULL CONTEXT PATTERNS")
    print("=" * 50)
    full_patterns = Counter()
    for p in all_patterns:
        if p['before'] and p['after']:
            # Use last word before and first word after
            before_words = p['before'].split()
            after_words = p['after'].split()
            if before_words and after_words:
                key = f"{before_words[-1]} {p['punct']} {after_words[0]}"
                full_patterns[key] += 1

    for pattern, count in full_patterns.most_common(30):
        print(f"  {pattern}: {count}")

    # Two-word before patterns (more specific)
    print("\n" + "=" * 50)
    print("TOP 20 TWO-WORD BEFORE PATTERNS")
    print("=" * 50)
    two_before = Counter()
    for p in all_patterns:
        words = p['before'].split()
        if len(words) >= 2:
            key = f"{words[-2]} {words[-1]} → {p['punct']}"
            two_before[key] += 1

    for pattern, count in two_before.most_common(20):
        print(f"  {pattern}: {count}")

    # Common sentence-ending patterns
    print("\n" + "=" * 50)
    print("SENTENCE BOUNDARY PATTERNS")
    print("=" * 50)

    # Find patterns where after starts with capital/new sentence indicators
    sentence_starters = ['إن', 'أن', 'و', 'في', 'من', 'على', 'إلى', 'هذا', 'هذه', 'كما', 'لكن', 'ولكن', 'أما', 'ثم']
    sentence_boundary = Counter()

    for p in all_patterns:
        after_words = p['after'].split()
        if after_words and after_words[0] in sentence_starters:
            key = f"{p['punct']} → {after_words[0]}"
            sentence_boundary[key] += 1

    for pattern, count in sentence_boundary.most_common(20):
        print(f"  {pattern}: {count}")

    # Extract actionable rules
    print("\n" + "=" * 70)
    print("ACTIONABLE RULES FOR PUNCT INSERTION")
    print("=" * 70)

    # High-confidence insertion rules
    print("\nRule type: INSERT punct after specific words")
    threshold = 50  # Minimum occurrences
    for pattern, count in before_patterns.most_common(50):
        if count >= threshold:
            word, punct = pattern.split(' → ')
            # Calculate what % of time this word has this punct after it
            # Would need to count total occurrences of word
            print(f"  {pattern}: {count} occurrences")

    print("\nRule type: INSERT punct before specific words")
    for pattern, count in after_patterns.most_common(50):
        if count >= threshold:
            print(f"  {pattern}: {count} occurrences")


if __name__ == "__main__":
    main()
