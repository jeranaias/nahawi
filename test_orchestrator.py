#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test the Nahawi Ensemble Orchestrator.

Tests:
1. Rule-based models work correctly
2. Cascading strategy applies models in sequence
3. Parallel strategy combines corrections
4. Specialist strategy routes to right models
"""

import sys
import io
import os

# Set encoding for Windows
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')

from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Add to path
sys.path.insert(0, str(Path('C:/nahawi')))

# Test cases
TEST_CASES = [
    # (input, expected_output, error_types)
    ("هذه مدرسه جميله", "هذه مدرسة جميلة", ["taa_marbuta"]),
    ("ذهب الي المدرسه", "ذهب إلى المدرسة", ["alif_maksura", "taa_marbuta"]),
    ("مرحبا, كيف حالك?", "مرحبا، كيف حالك؟", ["punctuation"]),
    ("ذهبت الى الى المدرسة", "ذهبت الى المدرسة", ["repeated_word"]),
]


def test_rule_based_direct():
    """Test rule-based models directly without orchestrator."""
    print("\n" + "=" * 60)
    print("TEST 1: Rule-Based Models (Direct)")
    print("=" * 60)

    from test_rules import TaaMarbutaFixer, AlifMaksuraFixer, PunctuationFixer, RepeatedWordFixer

    models = [
        ('TaaMarbutaFixer', TaaMarbutaFixer()),
        ('AlifMaksuraFixer', AlifMaksuraFixer()),
        ('PunctuationFixer', PunctuationFixer()),
        ('RepeatedWordFixer', RepeatedWordFixer()),
    ]

    passed = 0
    failed = 0

    for test_input, expected, _ in TEST_CASES:
        current = test_input
        applied = []

        for name, model in models:
            result, corrections = model.correct(current)
            if result != current:
                applied.append(name)
                current = result

        status = "PASS" if current == expected else "FAIL"
        if status == "PASS":
            passed += 1
        else:
            failed += 1

        print(f"\n[{status}] Input:    '{test_input}'")
        print(f"       Expected: '{expected}'")
        print(f"       Got:      '{current}'")
        print(f"       Applied:  {applied}")

    print(f"\nResult: {passed} passed, {failed} failed")
    return passed, failed


def test_orchestrator_import():
    """Test that orchestrator can be imported."""
    print("\n" + "=" * 60)
    print("TEST 2: Orchestrator Import")
    print("=" * 60)

    try:
        from nahawi_ensemble.orchestrator import NahawiEnsemble, EnsembleResult
        from nahawi_ensemble.config import config

        print("[PASS] Successfully imported NahawiEnsemble")
        print(f"       Config data_dir: {config.data_dir}")
        return True
    except Exception as e:
        print(f"[FAIL] Import error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_orchestrator_init():
    """Test orchestrator initialization with rule-based models only."""
    print("\n" + "=" * 60)
    print("TEST 3: Orchestrator Initialization")
    print("=" * 60)

    try:
        from nahawi_ensemble.orchestrator import NahawiEnsemble

        # Initialize with only rule-based models (no neural for now)
        # Note: names must match the keys in orchestrator._init_models()
        ensemble = NahawiEnsemble(
            enabled_models=['taa_marbuta_fixer', 'alif_maksura_fixer', 'punctuation_fixer', 'repeated_word_fixer'],
            lazy_load=True
        )

        print("[PASS] Orchestrator initialized")
        model_info = ensemble.get_model_info()
        print(f"       Available models: {list(model_info.keys())}")
        return ensemble
    except Exception as e:
        print(f"[FAIL] Init error: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_orchestrator_correction(ensemble):
    """Test orchestrator correction on test cases."""
    print("\n" + "=" * 60)
    print("TEST 4: Orchestrator Correction (Cascading)")
    print("=" * 60)

    if ensemble is None:
        print("[SKIP] Orchestrator not initialized")
        return 0, 0

    passed = 0
    failed = 0

    for test_input, expected, error_types in TEST_CASES:
        try:
            result = ensemble.correct(test_input, strategy="cascading")
            output = result.corrected_text

            status = "PASS" if output == expected else "FAIL"
            if status == "PASS":
                passed += 1
            else:
                failed += 1

            print(f"\n[{status}] Input:    '{test_input}'")
            print(f"       Expected: '{expected}'")
            print(f"       Got:      '{output}'")
            print(f"       Models:   {list(result.model_contributions.keys())}")

        except Exception as e:
            failed += 1
            print(f"\n[FAIL] Input: '{test_input}'")
            print(f"       Error: {e}")

    print(f"\nResult: {passed} passed, {failed} failed")
    return passed, failed


def test_orchestrator_strategies(ensemble):
    """Test different orchestrator strategies."""
    print("\n" + "=" * 60)
    print("TEST 5: Orchestrator Strategies")
    print("=" * 60)

    if ensemble is None:
        print("[SKIP] Orchestrator not initialized")
        return

    test_input = "هذه مدرسه جميله"
    expected = "هذه مدرسة جميلة"

    strategies = ["cascading", "parallel", "specialist"]

    for strategy in strategies:
        try:
            result = ensemble.correct(test_input, strategy=strategy)
            output = result.corrected_text
            status = "PASS" if output == expected else "FAIL"
            print(f"[{status}] Strategy '{strategy}': '{output}'")
        except Exception as e:
            print(f"[FAIL] Strategy '{strategy}': {e}")


def main():
    print("\n" + "=" * 60)
    print("NAHAWI ENSEMBLE ORCHESTRATOR TESTS")
    print("=" * 60)

    total_passed = 0
    total_failed = 0

    # Test 1: Direct rule-based
    p, f = test_rule_based_direct()
    total_passed += p
    total_failed += f

    # Test 2: Import
    if not test_orchestrator_import():
        print("\n[ABORT] Cannot continue without successful import")
        return

    # Test 3: Init
    ensemble = test_orchestrator_init()

    # Test 4: Corrections
    if ensemble:
        p, f = test_orchestrator_correction(ensemble)
        total_passed += p
        total_failed += f

    # Test 5: Strategies
    test_orchestrator_strategies(ensemble)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total: {total_passed} passed, {total_failed} failed")

    if total_failed == 0:
        print("All tests passed!")
    else:
        print(f"{total_failed} tests failed - review output above")

    print("=" * 60)


if __name__ == '__main__':
    main()
