#!/usr/bin/env python3
"""
Clean FASIH benchmark by removing non-GEC examples.

Non-GEC = word substitutions that aren't grammatical corrections:
- أعلن → رعت (announced → sponsored) - different verbs
- أعلن → افتتحت (announced → opened) - different verbs
- لتقييم → سريعة (for evaluation → fast) - unrelated words

Keep only examples where changes are:
1. Character-level edits (hamza, taa marbuta, etc.)
2. Morphological fixes (conjugation, case endings)
3. Spelling corrections (letter confusion)
4. Article addition/removal
"""

import json
from collections import defaultdict

FASIH_TEST = "/home/ubuntu/nahawi/data/fasih_test.json"
OUTPUT_PATH = "/home/ubuntu/nahawi/data/fasih_test_clean.json"

def levenshtein_ratio(s1, s2):
    """Compute similarity ratio between two strings."""
    if not s1 or not s2:
        return 0.0

    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0

    # Simple edit distance
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)

    distance = dp[len1][len2]
    max_len = max(len1, len2)
    return 1.0 - (distance / max_len)


def is_gec_correction(src_word, tgt_word):
    """
    Determine if src→tgt is a valid GEC correction.
    Returns (is_valid, reason)
    """
    if src_word == tgt_word:
        return True, "no_change"

    # Compute similarity
    sim = levenshtein_ratio(src_word, tgt_word)

    # High similarity (>0.5) = likely character-level fix
    if sim >= 0.5:
        return True, "char_level"

    # Check for specific patterns even with low similarity

    # Article addition: مركز → المركز
    if tgt_word == 'ال' + src_word or src_word == 'ال' + tgt_word:
        return True, "article"

    # Feminine ending: رئيسي → رئيسية
    if src_word + 'ة' == tgt_word or src_word + 'ه' == tgt_word:
        return True, "feminine"

    # Verb conjugation: يتعلموا → يتعلمون
    if src_word[:-2] == tgt_word[:-2] and len(src_word) > 3:
        return True, "conjugation"

    # If similarity is very low (<0.3) and no pattern match, it's likely semantic
    if sim < 0.3:
        return False, f"semantic_subst (sim={sim:.2f})"

    # Medium similarity - might be spelling
    return True, "spelling"


def find_word_changes(src, tgt):
    """Find word-level changes between source and target."""
    src_words = src.split()
    tgt_words = tgt.split()

    changes = []

    # Simple alignment by position
    max_len = max(len(src_words), len(tgt_words))

    for i in range(max_len):
        sw = src_words[i] if i < len(src_words) else ''
        tw = tgt_words[i] if i < len(tgt_words) else ''

        if sw != tw:
            is_valid, reason = is_gec_correction(sw, tw)
            changes.append({
                'src': sw,
                'tgt': tw,
                'is_gec': is_valid,
                'reason': reason
            })

    return changes


def main():
    print("=" * 70)
    print("CLEANING FASIH BENCHMARK")
    print("=" * 70)

    with open(FASIH_TEST, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} examples")

    clean_data = []
    removed_examples = []
    stats = defaultdict(int)

    for item in data:
        src = item.get('source', item.get('src', ''))
        tgt = item.get('target', item.get('tgt', ''))

        changes = find_word_changes(src, tgt)

        # Count non-GEC changes
        non_gec_changes = [c for c in changes if not c['is_gec']]

        if len(non_gec_changes) == 0:
            # All changes are valid GEC
            clean_data.append(item)
            stats['kept'] += 1
        elif len(non_gec_changes) == 1 and len(changes) > 3:
            # One bad change among many good ones - keep but note
            clean_data.append(item)
            stats['kept_with_noise'] += 1
        else:
            # Too many non-GEC changes - remove
            removed_examples.append({
                'src': src,
                'tgt': tgt,
                'non_gec_changes': non_gec_changes
            })
            stats['removed'] += 1

    print(f"\nResults:")
    print(f"  Kept (clean):      {stats['kept']}")
    print(f"  Kept (with noise): {stats['kept_with_noise']}")
    print(f"  Removed:           {stats['removed']}")
    print(f"  Total clean:       {len(clean_data)}")

    # Save clean benchmark
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(clean_data, f, ensure_ascii=False, indent=2)
    print(f"\nSaved clean benchmark to {OUTPUT_PATH}")

    # Show removed examples
    print("\n" + "=" * 70)
    print("REMOVED EXAMPLES (first 20)")
    print("=" * 70)

    for i, ex in enumerate(removed_examples[:20]):
        print(f"\n{i+1}. Non-GEC changes:")
        for c in ex['non_gec_changes']:
            print(f"   '{c['src']}' → '{c['tgt']}' ({c['reason']})")
        print(f"   Full src: {ex['src'][:80]}...")

    # Analyze what was removed
    print("\n" + "=" * 70)
    print("REMOVAL REASONS")
    print("=" * 70)

    reason_counts = defaultdict(int)
    for ex in removed_examples:
        for c in ex['non_gec_changes']:
            reason_counts[c['reason']] += 1

    for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count}")


if __name__ == "__main__":
    main()
