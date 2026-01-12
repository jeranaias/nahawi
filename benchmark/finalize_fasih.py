#!/usr/bin/env python3
"""
Finalize FASIH benchmark by combining all sources.
"""

import json
import random
import sys
from pathlib import Path
from collections import defaultdict, Counter

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

BENCHMARK_DIR = Path(__file__).parent
FASIH_DIR = BENCHMARK_DIR / "fasih"

# Target per category for Core
CORE_TARGET = 150

# Category mappings
CATEGORY_NORMALIZE = {
    'letter_dad_za': 'dad_za',
    'letter_dal_thal': 'dal_thal',
}


def load_json(path: Path) -> list:
    if not path.exists():
        return []
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(data)} samples to {path}")


def normalize_category(cat: str) -> str:
    return CATEGORY_NORMALIZE.get(cat, cat)


def deduplicate(samples: list) -> list:
    """Remove duplicate sentences."""
    seen = set()
    unique = []
    for s in samples:
        key = s.get('source', s.get('text', ''))
        if key not in seen:
            seen.add(key)
            unique.append(s)
    return unique


def main():
    print("=" * 60)
    print("FINALIZING FASIH BENCHMARK")
    print("=" * 60)

    # Load all sources
    print("\n=== Loading sources ===")

    # 1. Hunted samples (balanced, from corpus)
    hunted = load_json(FASIH_DIR / "hunted_samples.json")
    print(f"Hunted: {len(hunted)}")

    # 2. Existing core samples (from v5.1)
    existing_core = load_json(FASIH_DIR / "core" / "test.json")
    print(f"Existing core: {len(existing_core)}")

    # 3. Existing grammar samples (from QALB)
    existing_full = load_json(FASIH_DIR / "full" / "test.json")
    # Filter to just grammar categories
    grammar_cats = {'gender_agreement', 'number_agreement', 'verb_agreement',
                    'wrong_prep', 'missing_prep', 'definiteness', 'spelling', 'punctuation'}
    existing_grammar = [s for s in existing_full if s['category'] in grammar_cats]
    print(f"Existing grammar: {len(existing_grammar)}")

    # 4. Identity samples
    identity = load_json(FASIH_DIR / "identity" / "test.json")
    print(f"Identity: {len(identity)}")

    # Combine and balance core categories
    print("\n=== Building Core ===")

    core_by_cat = defaultdict(list)

    # Add hunted samples first (preferred - from corpus)
    for s in hunted:
        cat = normalize_category(s['category'])
        if cat in {'hamza', 'taa_marbuta', 'alif_maqsura', 'alif_madda', 'dad_za', 'dal_thal'}:
            core_by_cat[cat].append(s)

    # Top up with existing samples if needed
    for s in existing_core:
        cat = normalize_category(s['category'])
        if len(core_by_cat[cat]) < CORE_TARGET:
            core_by_cat[cat].append(s)

    # Balance to target
    core_samples = []
    for cat in sorted(core_by_cat.keys()):
        samples = core_by_cat[cat]
        random.shuffle(samples)
        selected = samples[:CORE_TARGET]
        core_samples.extend(selected)
        print(f"  {cat}: {len(selected)}")

    # Deduplicate
    core_samples = deduplicate(core_samples)

    # Reassign IDs
    for i, s in enumerate(core_samples):
        s['id'] = f"core-{s['category']}-{i:04d}"

    # Split into test/dev (85/15)
    random.shuffle(core_samples)
    split_idx = int(len(core_samples) * 0.85)
    core_test = core_samples[:split_idx]
    core_dev = core_samples[split_idx:]

    print(f"\nCore Test: {len(core_test)}")
    print(f"Core Dev: {len(core_dev)}")

    # Build Full (Core + Grammar)
    print("\n=== Building Full ===")

    # Add grammar samples
    grammar_by_cat = defaultdict(list)
    for s in existing_grammar:
        grammar_by_cat[s['category']].append(s)

    # Also add missing_prep from hunted
    for s in hunted:
        if s['category'] == 'missing_prep':
            grammar_by_cat['missing_prep'].append(s)

    # Balance grammar categories
    grammar_samples = []
    for cat in sorted(grammar_by_cat.keys()):
        samples = grammar_by_cat[cat]
        random.shuffle(samples)
        selected = samples[:120]  # 120 per grammar category
        grammar_samples.extend(selected)
        print(f"  {cat}: {len(selected)}")

    # Combine with core
    full_samples = core_samples + grammar_samples
    full_samples = deduplicate(full_samples)

    # Reassign IDs
    for i, s in enumerate(full_samples):
        s['id'] = f"full-{s['category']}-{i:04d}"

    # Split
    random.shuffle(full_samples)
    split_idx = int(len(full_samples) * 0.85)
    full_test = full_samples[:split_idx]
    full_dev = full_samples[split_idx:]

    print(f"\nFull Test: {len(full_test)}")
    print(f"Full Dev: {len(full_dev)}")

    # Save everything
    print("\n=== Saving ===")

    save_json(core_test, FASIH_DIR / "core" / "test.json")
    save_json(core_dev, FASIH_DIR / "core" / "dev.json")
    save_json(full_test, FASIH_DIR / "full" / "test.json")
    save_json(full_dev, FASIH_DIR / "full" / "dev.json")

    # Identity stays the same
    print(f"Identity: {len(identity)} samples (unchanged)")

    # Generate final rubric
    rubric = generate_rubric(core_test, full_test, identity)
    save_json(rubric, FASIH_DIR / "rubric.json")

    # Final summary
    print("\n" + "=" * 60)
    print("FASIH BENCHMARK FINALIZED")
    print("=" * 60)

    print("\n📊 FASIH-Core (Orthographic):")
    core_dist = Counter(s['category'] for s in core_test)
    for cat, count in sorted(core_dist.items()):
        print(f"   {cat}: {count}")
    print(f"   TOTAL: {len(core_test)} test + {len(core_dev)} dev")

    print("\n📊 FASIH-Full (Complete):")
    full_dist = Counter(s['category'] for s in full_test)
    for cat, count in sorted(full_dist.items()):
        print(f"   {cat}: {count}")
    print(f"   TOTAL: {len(full_test)} test + {len(full_dev)} dev")

    print("\n📊 FASIH-Identity (False Positive):")
    print(f"   TOTAL: {len(identity)} samples")

    total = len(core_test) + len(core_dev) + len(full_test) + len(full_dev) + len(identity)
    print(f"\n🎯 GRAND TOTAL: {total} samples")


def generate_rubric(core: list, full: list, identity: list) -> dict:
    """Generate the final rubric."""
    core_dist = Counter(s['category'] for s in core)
    full_dist = Counter(s['category'] for s in full)

    return {
        "name": "FASIH Arabic GEC Benchmark",
        "version": "1.0.0",
        "description": "World-class Arabic Grammatical Error Correction benchmark from real MSA text",
        "license": "CC-BY-4.0",
        "url": "https://github.com/nahawi/benchmark",

        "benchmarks": {
            "core": {
                "description": "Orthographic errors from real MSA Wikipedia/news text",
                "test_samples": len(core),
                "categories": sorted(core_dist.keys()),
                "distribution": dict(core_dist),
                "source": "MSA Corpus (Wikipedia, UN, News)"
            },
            "full": {
                "description": "Complete grammar coverage including morphology and syntax",
                "test_samples": len(full),
                "categories": sorted(full_dist.keys()),
                "distribution": dict(full_dist),
                "source": "MSA Corpus + QALB Real Data"
            },
            "identity": {
                "description": "Correct sentences for false positive testing",
                "test_samples": len(identity),
                "purpose": "Models should return these unchanged",
                "source": "MSA Corpus (clean samples)"
            }
        },

        "categories": {
            "hamza": {
                "name": "Hamza (همزة)",
                "description": "Missing or incorrect hamza placement",
                "examples": ["اعلان → إعلان", "اكثر → أكثر"]
            },
            "taa_marbuta": {
                "name": "Taa Marbuta (تاء مربوطة)",
                "description": "Confusion between ة and ه",
                "examples": ["المدرسه → المدرسة"]
            },
            "alif_maqsura": {
                "name": "Alif Maqsura (ألف مقصورة)",
                "description": "Confusion between ى and ي",
                "examples": ["علي → على", "إلي → إلى"]
            },
            "alif_madda": {
                "name": "Alif Madda (ألف المد)",
                "description": "Missing madda on alif",
                "examples": ["الان → الآن", "اخر → آخر"]
            },
            "dad_za": {
                "name": "Dad/Za (ض/ظ)",
                "description": "Confusion between emphatic consonants",
                "examples": ["نضر → نظر", "ضهر → ظهر"]
            },
            "dal_thal": {
                "name": "Dal/Thal (د/ذ)",
                "description": "Confusion between dental consonants",
                "examples": ["هدا → هذا", "كدلك → كذلك"]
            },
            "gender_agreement": {
                "name": "Gender Agreement",
                "description": "Noun-adjective gender mismatch",
                "examples": ["المدينة الكبير → المدينة الكبيرة"]
            },
            "number_agreement": {
                "name": "Number Agreement",
                "description": "Singular/plural mismatch",
                "examples": ["الطلاب الناجح → الطلاب الناجحون"]
            },
            "verb_agreement": {
                "name": "Verb Agreement",
                "description": "Subject-verb agreement errors",
                "examples": ["الطالبات درسوا → الطالبات درسن"]
            },
            "missing_prep": {
                "name": "Missing Preposition",
                "description": "Required preposition is missing",
                "examples": ["يبحث المعلومات → يبحث عن المعلومات"]
            },
            "wrong_prep": {
                "name": "Wrong Preposition",
                "description": "Incorrect preposition choice",
                "examples": ["يعتمد من → يعتمد على"]
            },
            "definiteness": {
                "name": "Definiteness",
                "description": "Article agreement errors",
                "examples": ["كتاب الجديد → الكتاب الجديد"]
            },
            "spelling": {
                "name": "Spelling",
                "description": "Common spelling errors",
                "examples": ["الإقتصاد → الاقتصاد"]
            },
            "punctuation": {
                "name": "Punctuation",
                "description": "Punctuation errors",
                "examples": ["Missing comma, wrong placement"]
            }
        },

        "evaluation": {
            "primary_metric": "F0.5",
            "rationale": "Precision matters more than recall - don't introduce new errors",
            "pass_threshold": 0.90,
            "minimum_threshold": 0.80,
            "false_positive_target": 0.05
        }
    }


if __name__ == "__main__":
    random.seed(42)
    main()
