#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick test of available ensemble models.
Tests whatever models are currently available.
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from pathlib import Path
import json

# Test sentences with common Arabic errors
TEST_CASES = [
    # Taa marbuta errors (rule-based)
    ("هذه مدرسه جميله", "هذه مدرسة جميلة"),

    # Alif maksura errors (rule-based)
    ("ذهبت الي المدرسه", "ذهبت إلى المدرسة"),

    # Punctuation (rule-based)
    ("مرحبا, كيف حالك?", "مرحبا، كيف حالك؟"),

    # Repeated word (rule-based)
    ("ذهبت الى الى المدرسة", "ذهبت الى المدرسة"),

    # Hamza errors (need neural)
    ("انا احب القراءه", "أنا أحب القراءة"),

    # Space errors (need neural)
    ("ذهبتالى المدرسة", "ذهبت إلى المدرسة"),

    # Spelling errors (need neural)
    ("الكتب المفيده", "الكتب المفيدة"),
]


def test_rule_based():
    """Test rule-based models directly."""
    print("=" * 60)
    print("TESTING RULE-BASED MODELS")
    print("=" * 60)

    # Import directly
    sys.path.insert(0, str(Path('C:/nahawi')))

    try:
        from test_rules import TaaMarbutaFixer, AlifMaksuraFixer, PunctuationFixer, RepeatedWordFixer

        models = [
            ('TaaMarbutaFixer', TaaMarbutaFixer()),
            ('AlifMaksuraFixer', AlifMaksuraFixer()),
            ('PunctuationFixer', PunctuationFixer()),
            ('RepeatedWordFixer', RepeatedWordFixer()),
        ]

        for test_input, expected in TEST_CASES[:4]:  # First 4 are rule-based
            print(f"\nInput:    '{test_input}'")
            print(f"Expected: '{expected}'")

            current = test_input
            applied = []
            for name, model in models:
                result, corrections = model.correct(current)
                if result != current:
                    applied.append(name)
                    current = result

            print(f"Output:   '{current}'")
            print(f"Applied:  {applied}")
            status = "PASS" if current == expected else "FAIL"
            print(f"Status:   [{status}]")

    except Exception as e:
        print(f"Error: {e}")


def test_neural_models():
    """Test neural models if available."""
    print("\n" + "=" * 60)
    print("TESTING NEURAL MODELS")
    print("=" * 60)

    checkpoint_dir = Path('C:/nahawi/nahawi_ensemble/checkpoints')

    models_available = []
    for model_name in ['general_gec', 'hamza_fixer', 'space_fixer', 'spelling_fixer']:
        final_dir = checkpoint_dir / model_name / 'final'
        if final_dir.exists() and (final_dir / 'model.safetensors').exists():
            models_available.append(model_name)

    if not models_available:
        print("\nNo trained neural models available yet.")
        print("Run training scripts and check back later.")
        return

    print(f"\nAvailable models: {models_available}")

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        for model_name in models_available:
            print(f"\n--- Testing {model_name} ---")

            model_path = checkpoint_dir / model_name / 'final'
            tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            model = AutoModelForSeq2SeqLM.from_pretrained(str(model_path))
            model.eval()

            # Test on a few examples
            for test_input, expected in TEST_CASES[-3:]:  # Last 3 need neural
                inputs = tokenizer(test_input, return_tensors="pt", padding=True, truncation=True)

                with torch.no_grad():
                    outputs = model.generate(**inputs, max_length=128, num_beams=4)

                result = tokenizer.decode(outputs[0], skip_special_tokens=True)

                print(f"\n  Input:  '{test_input}'")
                print(f"  Output: '{result}'")
                print(f"  Expect: '{expected}'")

    except Exception as e:
        print(f"Error testing neural models: {e}")


def main():
    print("\nNAHAWI ENSEMBLE - QUICK TEST\n")

    test_rule_based()
    test_neural_models()

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
