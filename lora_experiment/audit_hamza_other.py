#!/usr/bin/env python3
"""Audit FASIH hamza_other category - what's actually in there?"""

import json

FASIH_TEST = "/home/ubuntu/nahawi/data/fasih_test.json"

def classify_error_type(src_word, tgt_word):
    """Same classification as error_analysis.py"""
    if src_word == tgt_word:
        return None

    hamza_chars = set('أإآؤئء')

    # Check for hamza-related changes
    src_hamza = set(c for c in src_word if c in hamza_chars)
    tgt_hamza = set(c for c in tgt_word if c in hamza_chars)

    # Specific hamza patterns
    if 'إ' in tgt_word and 'ا' in src_word and src_word.replace('ا', 'إ', 1) == tgt_word:
        return 'hamza_add_alif_kasra'
    if 'أ' in tgt_word and 'ا' in src_word and src_word.replace('ا', 'أ', 1) == tgt_word:
        return 'hamza_add_alif'
    if 'آ' in tgt_word and 'ا' in src_word:
        return 'alif_madda'
    if 'ؤ' in tgt_word and 'و' in src_word:
        return 'hamza_waw'
    if 'ئ' in tgt_word and 'ي' in src_word:
        return 'hamza_yaa'
    if 'ء' in tgt_word and 'ء' not in src_word:
        return 'hamza_standalone'

    # Hamza swap
    if src_hamza and tgt_hamza and src_hamza != tgt_hamza:
        return 'hamza_alif_swap'

    # Generic hamza_other - catch-all for hamza-related
    if src_hamza != tgt_hamza or any(c in src_word + tgt_word for c in hamza_chars):
        return 'hamza_other'

    # Taa marbuta
    if ('ه' in src_word and 'ة' in tgt_word) or ('ة' in src_word and 'ه' in tgt_word):
        return 'taa_marbuta'

    # Alif maqsura
    if ('ي' in src_word and 'ى' in tgt_word) or ('ى' in src_word and 'ي' in tgt_word):
        return 'alif_maqsura'

    # Length-based
    len_diff = len(tgt_word) - len(src_word)
    if len_diff == 1:
        return 'one_char_del'
    if len_diff == -1:
        return 'one_char_ins'
    if len_diff > 1:
        return 'multi_char_del'
    if len_diff < -1:
        return 'multi_char_ins'

    # Same length = substitution
    if len(src_word) == len(tgt_word):
        diff_count = sum(1 for a, b in zip(src_word, tgt_word) if a != b)
        if diff_count == 1:
            return 'single_char_subst'
        else:
            return 'multi_char_subst'

    return 'other'


def main():
    print("=" * 80)
    print("FASIH hamza_other AUDIT")
    print("=" * 80)

    with open(FASIH_TEST, 'r', encoding='utf-8') as f:
        data = json.load(f)

    hamza_other_examples = []

    for item in data:
        src = item.get('source', item.get('src', ''))
        tgt = item.get('target', item.get('tgt', ''))

        src_words = src.split()
        tgt_words = tgt.split()

        # Align words (simple positional)
        for i, (sw, tw) in enumerate(zip(src_words, tgt_words)):
            if sw != tw:
                error_type = classify_error_type(sw, tw)
                if error_type == 'hamza_other':
                    hamza_other_examples.append({
                        'src_word': sw,
                        'tgt_word': tw,
                        'context_src': ' '.join(src_words[max(0,i-2):i+3]),
                        'context_tgt': ' '.join(tgt_words[max(0,i-2):i+3]),
                    })

    print(f"\nFound {len(hamza_other_examples)} hamza_other examples")
    print("\n" + "=" * 80)
    print("FIRST 30 EXAMPLES:")
    print("=" * 80)

    # Categorize what we find
    verb_ending = 0
    true_hamza = 0
    other = 0

    for i, ex in enumerate(hamza_other_examples[:30]):
        sw, tw = ex['src_word'], ex['tgt_word']

        # Detect pattern
        if sw.endswith('وا') and tw.endswith('ون'):
            pattern = "VERB ENDING (وا→ون)"
            verb_ending += 1
        elif sw.endswith('وا') and tw.endswith('وه'):
            pattern = "VERB ENDING (وا→وه)"
            verb_ending += 1
        elif any(c in sw + tw for c in 'أإآؤئء'):
            pattern = "TRUE HAMZA"
            true_hamza += 1
        else:
            pattern = "OTHER/UNKNOWN"
            other += 1

        print(f"\n{i+1}. '{sw}' → '{tw}'")
        print(f"   Pattern: {pattern}")
        print(f"   Context: {ex['context_src']}")

    print("\n" + "=" * 80)
    print("SUMMARY (first 30):")
    print("=" * 80)
    print(f"  Verb endings (وا→ون/وه): {verb_ending} ({100*verb_ending/30:.1f}%)")
    print(f"  True hamza errors:       {true_hamza} ({100*true_hamza/30:.1f}%)")
    print(f"  Other/unknown:           {other} ({100*other/30:.1f}%)")

    if verb_ending > true_hamza:
        print("\n⚠️  CONCLUSION: hamza_other is mostly VERB CONJUGATION, not hamza!")
        print("   Need to generate verb ending training data (وا→ون patterns)")


if __name__ == "__main__":
    main()
