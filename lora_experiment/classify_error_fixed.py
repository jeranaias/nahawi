#!/usr/bin/env python3
"""
Fixed error classifier: Categorize by WHAT CHANGED, not what's in the word.

Priority order:
1. taa_marbuta (ة↔ه)
2. Letter confusions (د↔ذ, ض↔ظ, س↔ص, etc.)
3. Verb conjugation (وا→ون, gender agreement)
4. Alif maqsura (ى↔ي)
5. Hamza changes (only if actual hamza is the change)
6. Length-based (insertions/deletions)
7. Generic substitution
"""

def get_char_diff(src, tgt):
    """Find the character-level differences between two words."""
    # Simple: find positions where chars differ
    diffs = []
    min_len = min(len(src), len(tgt))

    for i in range(min_len):
        if src[i] != tgt[i]:
            diffs.append((i, src[i], tgt[i]))

    # Handle length differences
    if len(src) > len(tgt):
        for i in range(min_len, len(src)):
            diffs.append((i, src[i], None))  # deleted
    elif len(tgt) > len(src):
        for i in range(min_len, len(tgt)):
            diffs.append((i, None, tgt[i]))  # inserted

    return diffs


def classify_error_fixed(src_word, tgt_word):
    """
    Classify error by WHAT CHANGED, not by what else is in the word.
    Returns (error_type, description)
    """
    if src_word == tgt_word:
        return None, None

    src = src_word.strip()
    tgt = tgt_word.strip()

    # Get character differences
    diffs = get_char_diff(src, tgt)

    if not diffs:
        return None, None

    # Extract the changed characters
    changed_from = set(d[1] for d in diffs if d[1])
    changed_to = set(d[2] for d in diffs if d[2])

    # =================================================================
    # PRIORITY 1: Taa marbuta (ة↔ه)
    # =================================================================
    if ('ه' in changed_from and 'ة' in changed_to) or ('ة' in changed_from and 'ه' in changed_to):
        return 'taa_marbuta', f"ه↔ة: {src}→{tgt}"

    # =================================================================
    # PRIORITY 2: Letter confusions
    # =================================================================
    confusion_pairs = [
        ('د', 'ذ', 'dal_thal_confusion'),
        ('ذ', 'د', 'dal_thal_confusion'),
        ('ض', 'ظ', 'dad_za_confusion'),
        ('ظ', 'ض', 'dad_za_confusion'),
        ('س', 'ص', 'seen_sad_confusion'),
        ('ص', 'س', 'seen_sad_confusion'),
        ('ت', 'ط', 'taa_ta_confusion'),
        ('ط', 'ت', 'taa_ta_confusion'),
    ]

    for c1, c2, err_type in confusion_pairs:
        if c1 in changed_from and c2 in changed_to:
            return err_type, f"{c1}→{c2}: {src}→{tgt}"

    # =================================================================
    # PRIORITY 3: Verb conjugation patterns
    # =================================================================
    # وا → ون (plural masculine verb ending)
    if src.endswith('وا') and tgt.endswith('ون'):
        return 'verb_conjugation', f"وا→ون: {src}→{tgt}"
    if src.endswith('ون') and tgt.endswith('وا'):
        return 'verb_conjugation', f"ون→وا: {src}→{tgt}"

    # Gender agreement: ى → ت (feminine verb)
    if src.endswith('ى') and tgt.endswith('ت') and len(src) == len(tgt):
        return 'verb_gender', f"ى→ت: {src}→{tgt}"
    if src.endswith('ت') and tgt.endswith('ى') and len(src) == len(tgt):
        return 'verb_gender', f"ت→ى: {src}→{tgt}"

    # =================================================================
    # PRIORITY 4: Alif maqsura (ى↔ي) - but NOT verb gender
    # =================================================================
    if ('ي' in changed_from and 'ى' in changed_to) or ('ى' in changed_from and 'ي' in changed_to):
        # Make sure it's not verb gender (handled above)
        if not (src.endswith('ى') and tgt.endswith('ت')):
            return 'alif_maqsura', f"ى↔ي: {src}→{tgt}"

    # =================================================================
    # PRIORITY 5: Hamza changes (only if hamza IS the change)
    # =================================================================
    hamza_chars = set('أإآؤئء')

    # Check if hamza is actually what changed
    hamza_in_change = (changed_from & hamza_chars) or (changed_to & hamza_chars)
    alif_involved = 'ا' in changed_from or 'ا' in changed_to
    waw_involved = 'و' in changed_from or 'و' in changed_to
    yaa_involved = 'ي' in changed_from or 'ي' in changed_to

    if hamza_in_change or (alif_involved and (changed_to & hamza_chars)):
        # Specific hamza types
        if 'ا' in changed_from and 'إ' in changed_to:
            return 'hamza_add_alif_kasra', f"ا→إ: {src}→{tgt}"
        if 'ا' in changed_from and 'أ' in changed_to:
            return 'hamza_add_alif', f"ا→أ: {src}→{tgt}"
        if 'ا' in changed_from and 'آ' in changed_to:
            return 'alif_madda', f"ا→آ: {src}→{tgt}"
        if 'و' in changed_from and 'ؤ' in changed_to:
            return 'hamza_waw', f"و→ؤ: {src}→{tgt}"
        if 'ي' in changed_from and 'ئ' in changed_to:
            return 'hamza_yaa', f"ي→ئ: {src}→{tgt}"
        if 'ء' in changed_to and 'ء' not in changed_from:
            return 'hamza_standalone', f"→ء: {src}→{tgt}"

        # Hamza position swap
        if (changed_from & hamza_chars) and (changed_to & hamza_chars):
            return 'hamza_swap', f"hamza swap: {src}→{tgt}"

        # Generic hamza (actually involves hamza change)
        return 'hamza_other', f"hamza: {src}→{tgt}"

    # =================================================================
    # PRIORITY 6: Length-based (insertions/deletions)
    # =================================================================
    len_diff = len(tgt) - len(src)

    if len_diff == 1:
        return 'one_char_del', f"missing char: {src}→{tgt}"
    if len_diff == -1:
        return 'one_char_ins', f"extra char: {src}→{tgt}"
    if len_diff > 1:
        return 'multi_char_del', f"missing {len_diff} chars: {src}→{tgt}"
    if len_diff < -1:
        return 'multi_char_ins', f"extra {-len_diff} chars: {src}→{tgt}"

    # =================================================================
    # PRIORITY 7: Same length = substitution
    # =================================================================
    if len(src) == len(tgt):
        diff_count = sum(1 for a, b in zip(src, tgt) if a != b)
        if diff_count == 1:
            return 'single_char_subst', f"1 char diff: {src}→{tgt}"
        else:
            return 'multi_char_subst', f"{diff_count} chars diff: {src}→{tgt}"

    return 'other', f"unknown: {src}→{tgt}"


def test_classifier():
    """Test with known examples."""
    test_cases = [
        # Taa marbuta (should be priority 1)
        ('الإسكندريه', 'الإسكندرية', 'taa_marbuta'),
        ('إيجابيه', 'إيجابية', 'taa_marbuta'),
        ('عائشه', 'عائشة', 'taa_marbuta'),
        ('رؤيه', 'رؤية', 'taa_marbuta'),

        # Letter confusion (priority 2)
        ('أكذ', 'أكد', 'dal_thal_confusion'),
        ('نهظ', 'نهض', 'dad_za_confusion'),

        # Verb conjugation (priority 3)
        ('يتعلموا', 'يتعلمون', 'verb_conjugation'),
        ('يعالجوا', 'يعالجون', 'verb_conjugation'),

        # Verb gender (priority 3)
        ('أنهى', 'أنهت', 'verb_gender'),
        ('أطلق', 'أطلقت', 'one_char_del'),  # This is insertion, not gender

        # Alif maqsura (priority 4)
        ('علي', 'على', 'alif_maqsura'),
        ('حتي', 'حتى', 'alif_maqsura'),

        # Hamza (priority 5) - only when hamza IS the change
        ('الي', 'إلى', 'hamza_add_alif_kasra'),
        ('اهمية', 'أهمية', 'hamza_add_alif'),
        ('اخر', 'آخر', 'alif_madda'),
        ('مسوول', 'مسؤول', 'hamza_waw'),
    ]

    print("=" * 80)
    print("CLASSIFIER TEST")
    print("=" * 80)

    passed = 0
    failed = 0

    for src, tgt, expected in test_cases:
        result, desc = classify_error_fixed(src, tgt)
        status = "✓" if result == expected else "✗"
        if result == expected:
            passed += 1
        else:
            failed += 1
        print(f"{status} '{src}' → '{tgt}'")
        print(f"   Expected: {expected}, Got: {result}")
        if desc:
            print(f"   {desc}")

    print(f"\nPassed: {passed}/{len(test_cases)}, Failed: {failed}/{len(test_cases)}")


if __name__ == "__main__":
    test_classifier()
