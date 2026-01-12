#!/usr/bin/env python3
"""
Comprehensive tests for the Nahawi Arabic Morphology Tool.
Tests all components against the elite standard requirements.
"""

import sys
sys.path.insert(0, '..')

from nahawi_morphology import ArabicMorphology


def test_gender_detection():
    """Test gender detection accuracy."""
    print("\n" + "=" * 60)
    print("TESTING: Gender Detection")
    print("=" * 60)

    morph = ArabicMorphology()

    test_cases = [
        # (word, expected_gender)
        # Masculine (no marker)
        ("كاتب", "masc"),
        ("طالب", "masc"),
        ("كتاب", "masc"),
        ("قلم", "masc"),

        # Feminine (taa marbuta)
        ("كاتبة", "fem"),
        ("طالبة", "fem"),
        ("مدرسة", "fem"),
        ("جامعة", "fem"),

        # Inherently feminine
        ("شمس", "fem"),
        ("أرض", "fem"),
        ("أم", "fem"),

        # Masculine exceptions (despite ة)
        ("خليفة", "masc"),

        # With definite article
        ("الكاتب", "masc"),
        ("الكاتبة", "fem"),
        ("الطالبات", "fem"),

        # Plurals
        ("طلاب", "masc"),
        ("معلمون", "masc"),
        ("معلمات", "fem"),
    ]

    passed = 0
    for word, expected in test_cases:
        info = morph.analyze(word)
        actual = info.gender
        status = "✓" if actual == expected else "✗"
        if actual == expected:
            passed += 1
        print(f"  {status} {word}: expected={expected}, got={actual}")

    print(f"\nGender Detection: {passed}/{len(test_cases)} passed ({100*passed/len(test_cases):.1f}%)")
    return passed == len(test_cases)


def test_number_detection():
    """Test number detection accuracy."""
    print("\n" + "=" * 60)
    print("TESTING: Number Detection")
    print("=" * 60)

    morph = ArabicMorphology()

    test_cases = [
        # Singular
        ("طالب", "sing"),
        ("كتاب", "sing"),
        ("مدرسة", "sing"),

        # Plural (sound masculine)
        ("معلمون", "plural"),
        ("مهندسون", "plural"),

        # Plural (sound feminine)
        ("طالبات", "plural"),
        ("سيارات", "plural"),

        # Plural (broken)
        ("كتب", "plural"),
        ("طلاب", "plural"),
        ("رجال", "plural"),
        ("أيام", "plural"),

        # Dual
        ("طالبان", "dual"),
    ]

    passed = 0
    for word, expected in test_cases:
        info = morph.analyze(word)
        actual = info.number
        status = "✓" if actual == expected else "✗"
        if actual == expected:
            passed += 1
        print(f"  {status} {word}: expected={expected}, got={actual}")

    print(f"\nNumber Detection: {passed}/{len(test_cases)} passed ({100*passed/len(test_cases):.1f}%)")
    return passed == len(test_cases)


def test_agreement_checking():
    """Test agreement error detection."""
    print("\n" + "=" * 60)
    print("TESTING: Agreement Checking")
    print("=" * 60)

    morph = ArabicMorphology()

    test_cases = [
        # (word1, word2, should_have_error)
        # Gender errors
        ("الطالبة", "المجتهد", True),  # fem noun + masc adj
        ("الطالبة", "المجتهدة", False),  # fem noun + fem adj
        ("الطالب", "المجتهدة", True),  # masc noun + fem adj
        ("الطالب", "المجتهد", False),  # masc noun + masc adj

        # Verb agreement
        ("الطالبة", "نجح", True),  # fem noun + masc verb
        ("الطالبة", "نجحت", False),  # fem noun + fem verb
        ("الطالب", "نجحت", True),  # masc noun + fem verb
        ("الطالب", "نجح", False),  # masc noun + masc verb

        # With كان
        ("الشركة", "كان", True),  # fem + masc كان
        ("الشركة", "كانت", False),  # fem + fem كانت
    ]

    passed = 0
    for word1, word2, should_error in test_cases:
        errors = morph.check_agreement(word1, word2)
        has_error = len(errors) > 0
        status = "✓" if has_error == should_error else "✗"
        if has_error == should_error:
            passed += 1
        err_str = errors[0].error_type if errors else "none"
        print(f"  {status} {word1} + {word2}: expect_error={should_error}, got_error={has_error} ({err_str})")

    print(f"\nAgreement Checking: {passed}/{len(test_cases)} passed ({100*passed/len(test_cases):.1f}%)")
    return passed == len(test_cases)


def test_verb_conjugation():
    """Test verb conjugation."""
    print("\n" + "=" * 60)
    print("TESTING: Verb Conjugation")
    print("=" * 60)

    morph = ArabicMorphology()

    test_cases = [
        # (root, tense, person, expected)
        ("كتب", "past", "3ms", "كتب"),
        ("كتب", "past", "3fs", "كتبت"),
        ("كتب", "past", "3mp", "كتبوا"),
        ("كتب", "present", "3ms", "يكتب"),
        ("كتب", "present", "3fs", "تكتب"),
        ("كتب", "present", "3mp", "يكتبون"),

        # Irregular
        ("كون", "past", "3ms", "كان"),
        ("كون", "past", "3fs", "كانت"),
        ("كون", "past", "3mp", "كانوا"),
    ]

    passed = 0
    for root, tense, person, expected in test_cases:
        actual = morph.conjugate(root, tense, person)
        status = "✓" if actual == expected else "✗"
        if actual == expected:
            passed += 1
        print(f"  {status} {root}/{tense}/{person}: expected={expected}, got={actual}")

    print(f"\nVerb Conjugation: {passed}/{len(test_cases)} passed ({100*passed/len(test_cases):.1f}%)")
    return passed == len(test_cases)


def test_error_injection():
    """Test error injection for training data."""
    print("\n" + "=" * 60)
    print("TESTING: Error Injection")
    print("=" * 60)

    morph = ArabicMorphology()

    test_cases = [
        # (word, error_type, description)
        ("ذهبت", "gender", "Remove feminine marker"),
        ("كانت", "gender", "Remove feminine marker"),
        ("يدرسون", "number", "ون→وا confusion"),
        ("الوقت", "truncation", "Missing final ت"),
        ("المعلمين", "truncation", "Missing final ن"),
        ("إلى", "preposition", "Wrong preposition"),
        ("الكتاب", "definiteness", "Remove article"),
        ("هذا", "spelling", "Letter confusion"),
    ]

    passed = 0
    for word, error_type, desc in test_cases:
        result = morph.inject_error(word, error_type)
        # Check that we got a different result
        is_valid = result is not None and result != word
        status = "✓" if is_valid else "✗"
        if is_valid:
            passed += 1
        print(f"  {status} {word} ({error_type}): {word} → {result} ({desc})")

    print(f"\nError Injection: {passed}/{len(test_cases)} passed ({100*passed/len(test_cases):.1f}%)")
    return passed >= len(test_cases) * 0.8  # Allow 80% pass rate


def test_sentence_fixing():
    """Test sentence-level agreement fixing."""
    print("\n" + "=" * 60)
    print("TESTING: Sentence Agreement Fixing")
    print("=" * 60)

    morph = ArabicMorphology()

    test_cases = [
        # (input, expected_output_contains)
        ("الطالبة المجتهد", "المجتهدة"),
        ("المدينة الكبير", "الكبيرة"),
    ]

    passed = 0
    for inp, expected_contains in test_cases:
        result = morph.fix_agreement(inp)
        is_valid = expected_contains in result
        status = "✓" if is_valid else "✗"
        if is_valid:
            passed += 1
        print(f"  {status} '{inp}' → '{result}' (should contain '{expected_contains}')")

    print(f"\nSentence Fixing: {passed}/{len(test_cases)} passed ({100*passed/len(test_cases):.1f}%)")
    return passed == len(test_cases)


def test_performance():
    """Test performance requirements."""
    print("\n" + "=" * 60)
    print("TESTING: Performance")
    print("=" * 60)

    import time
    morph = ArabicMorphology()

    # Test words
    words = ["كتاب", "طالبة", "يذهبون", "المعلمات", "أيام"] * 200  # 1000 words

    # Analyze performance
    start = time.time()
    for word in words:
        morph.analyze(word)
    elapsed = time.time() - start

    per_word = elapsed / len(words) * 1000  # ms
    print(f"  Analysis: {len(words)} words in {elapsed:.3f}s ({per_word:.3f}ms/word)")

    passed = per_word < 1  # Target: <1ms per word
    print(f"  {'✓' if passed else '✗'} Target: <1ms/word, Actual: {per_word:.3f}ms/word")

    return passed


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "=" * 70)
    print("NAHAWI ARABIC MORPHOLOGY TOOL - TEST SUITE")
    print("=" * 70)

    results = {
        "Gender Detection": test_gender_detection(),
        "Number Detection": test_number_detection(),
        "Agreement Checking": test_agreement_checking(),
        "Verb Conjugation": test_verb_conjugation(),
        "Error Injection": test_error_injection(),
        "Sentence Fixing": test_sentence_fixing(),
        "Performance": test_performance(),
    }

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {test_name}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("ALL TESTS PASSED - Morphology tool meets elite standard!")
    else:
        print("SOME TESTS FAILED - Review and fix before proceeding.")
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    run_all_tests()
