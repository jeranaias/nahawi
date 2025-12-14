#!/usr/bin/env python3
"""
Quick test of rule-based models.

Run without any heavy dependencies to verify the rule-based models work.
"""

import sys
from pathlib import Path

# Add parent to path - import rule_based directly to avoid torch imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Direct import to avoid the __init__ chain that imports torch
import importlib.util
spec = importlib.util.spec_from_file_location(
    "rule_based",
    Path(__file__).parent.parent / "models" / "rule_based.py"
)
rule_based = importlib.util.module_from_spec(spec)

# We need base.py too
spec_base = importlib.util.spec_from_file_location(
    "base",
    Path(__file__).parent.parent / "models" / "base.py"
)
base = importlib.util.module_from_spec(spec_base)

# Mock torch for base.py
class MockTorch:
    class Tensor:
        pass
sys.modules['torch'] = MockTorch()

spec_base.loader.exec_module(base)
sys.modules['nahawi_ensemble.models.base'] = base

spec.loader.exec_module(rule_based)

TaaMarbutaFixer = rule_based.TaaMarbutaFixer
AlifMaksuraFixer = rule_based.AlifMaksuraFixer
PunctuationFixer = rule_based.PunctuationFixer
RepeatedWordFixer = rule_based.RepeatedWordFixer

def test_taa_marbuta():
    print("=" * 50)
    print("Testing TaaMarbutaFixer")
    print("=" * 50)

    model = TaaMarbutaFixer()

    tests = [
        ("هذه مدرسه جميله", "هذه مدرسة جميلة"),
        ("الجامعه الكبيره", "الجامعة الكبيرة"),
        ("حياه سعيده", "حياة سعيدة"),
        ("هذا بيت كبير", "هذا بيت كبير"),  # No change
    ]

    for input_text, expected in tests:
        result = model.correct(input_text)
        status = "PASS" if result.corrected_text == expected else "FAIL"
        print(f"[{status}] '{input_text}'")
        print(f"       -> '{result.corrected_text}'")
        print(f"       Expected: '{expected}'")
        if result.corrections:
            print(f"       Corrections: {len(result.corrections)}")
        print()


def test_alif_maksura():
    print("=" * 50)
    print("Testing AlifMaksuraFixer")
    print("=" * 50)

    model = AlifMaksuraFixer()

    tests = [
        ("ذهب الي المدرسه", "ذهب إلى المدرسه"),  # الي -> إلى
        ("متي سافرت", "متى سافرت"),
        ("حتي الان", "حتى الان"),
        ("هو قوي جدا", "هو قوي جدا"),  # No change
    ]

    for input_text, expected in tests:
        result = model.correct(input_text)
        status = "PASS" if result.corrected_text == expected else "FAIL"
        print(f"[{status}] '{input_text}'")
        print(f"       -> '{result.corrected_text}'")
        print(f"       Expected: '{expected}'")
        print()


def test_punctuation():
    print("=" * 50)
    print("Testing PunctuationFixer")
    print("=" * 50)

    model = PunctuationFixer()

    tests = [
        ("مرحبا, كيف حالك?", "مرحبا، كيف حالك؟"),
        ("هذا جيد; لكن ذاك افضل", "هذا جيد؛ لكن ذاك افضل"),
        ("مرحبا، كيف الحال؟", "مرحبا، كيف الحال؟"),  # Already correct
    ]

    for input_text, expected in tests:
        result = model.correct(input_text)
        status = "PASS" if result.corrected_text == expected else "FAIL"
        print(f"[{status}] '{input_text}'")
        print(f"       -> '{result.corrected_text}'")
        print(f"       Expected: '{expected}'")
        print()


def test_repeated_word():
    print("=" * 50)
    print("Testing RepeatedWordFixer")
    print("=" * 50)

    model = RepeatedWordFixer()

    tests = [
        ("ذهبت الى الى المدرسة", "ذهبت الى المدرسة"),
        ("هذا هذا جميل جدا", "هذا جميل جدا"),
        ("لا لا اريد ذلك", "لا لا اريد ذلك"),  # "لا لا" allowed for emphasis
    ]

    for input_text, expected in tests:
        result = model.correct(input_text)
        status = "PASS" if result.corrected_text == expected else "FAIL"
        print(f"[{status}] '{input_text}'")
        print(f"       -> '{result.corrected_text}'")
        print(f"       Expected: '{expected}'")
        print()


def main():
    print("\nNahawi Ensemble - Rule-Based Model Tests\n")

    test_taa_marbuta()
    test_alif_maksura()
    test_punctuation()
    test_repeated_word()

    print("=" * 50)
    print("Tests complete!")
    print("=" * 50)


if __name__ == '__main__':
    main()
