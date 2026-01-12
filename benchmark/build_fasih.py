#!/usr/bin/env python3
"""
Build FASIH Benchmark Suite

Creates:
- fasih-core: Orthographic errors from real MSA text
- fasih-full: Core + grammar/syntax from QALB
- fasih-identity: Correct sentences for false positive testing
"""

import json
import random
import re
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple

# Paths
BENCHMARK_DIR = Path(__file__).parent
PROJECT_ROOT = BENCHMARK_DIR.parent
FASIH_DIR = BENCHMARK_DIR / "fasih"

# Source files
V51_TEST = BENCHMARK_DIR / "fasih_v5.1" / "fasih_test.json"
V51_DEV = BENCHMARK_DIR / "fasih_v5.1" / "fasih_dev.json"
QALB_TRAIN = PROJECT_ROOT / "data" / "qalb_real_train.json"
MSA_CORPUS = PROJECT_ROOT / "corpus" / "msa_corpus_full.txt"

# Core categories (orthographic - from real corpus)
CORE_CATEGORIES = {
    'hamza', 'taa_marbuta', 'alif_maqsura', 'alif_madda',
    'letter_dad_za', 'letter_dal_thal'
}

# Renamed categories for clarity
CATEGORY_RENAME = {
    'letter_dad_za': 'dad_za',
    'letter_dal_thal': 'dal_thal',
}

# Grammar categories (from QALB)
GRAMMAR_CATEGORIES = {
    'gender_agreement', 'number_agreement', 'verb_agreement',
    'missing_prep', 'wrong_prep', 'definiteness', 'spelling', 'punctuation'
}


def load_json(path: Path) -> List[Dict]:
    """Load JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: List[Dict], path: Path):
    """Save JSON file with proper formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(data)} samples to {path}")


def extract_core_from_v51():
    """Extract orthographic samples from v5.1 (real corpus samples)."""
    print("\n=== Extracting FASIH-Core from v5.1 ===")

    test_data = load_json(V51_TEST)
    dev_data = load_json(V51_DEV)

    core_test = []
    core_dev = []

    stats = defaultdict(int)

    for i, sample in enumerate(test_data):
        error_type = sample.get('error_type', '')

        # Only keep orthographic categories (real corpus)
        if error_type in CORE_CATEGORIES:
            # Rename for clarity
            category = CATEGORY_RENAME.get(error_type, error_type)

            core_sample = {
                'id': f"core-{category}-{len([s for s in core_test if s['category'] == category]):04d}",
                'source': sample['source'],
                'target': sample['target'],
                'category': category,
                'correction': sample.get('correction', ''),
                'source_corpus': 'wikipedia',  # v5.1 real samples are from Wikipedia
                'difficulty': classify_difficulty(sample['source'], sample['target'])
            }
            core_test.append(core_sample)
            stats[category] += 1

    # Same for dev
    for sample in dev_data:
        error_type = sample.get('error_type', '')
        if error_type in CORE_CATEGORIES:
            category = CATEGORY_RENAME.get(error_type, error_type)
            core_sample = {
                'id': f"core-{category}-dev-{len([s for s in core_dev if s['category'] == category]):04d}",
                'source': sample['source'],
                'target': sample['target'],
                'category': category,
                'correction': sample.get('correction', ''),
                'source_corpus': 'wikipedia',
                'difficulty': classify_difficulty(sample['source'], sample['target'])
            }
            core_dev.append(core_sample)

    print(f"\nCore Test distribution:")
    for cat, count in sorted(stats.items()):
        print(f"  {cat}: {count}")
    print(f"  TOTAL: {len(core_test)}")

    return core_test, core_dev


def classify_difficulty(source: str, target: str) -> str:
    """Classify error difficulty based on edit distance and context."""
    # Count differences
    source_words = source.split()
    target_words = target.split()

    diff_count = sum(1 for s, t in zip(source_words, target_words) if s != t)
    diff_count += abs(len(source_words) - len(target_words))

    if diff_count <= 1:
        return 'easy'
    elif diff_count <= 3:
        return 'medium'
    else:
        return 'hard'


def classify_qalb_error(source: str, target: str) -> Tuple[str, str]:
    """
    Classify error type in a QALB pair.
    Returns (category, correction_description) or (None, None) if unclear.
    """
    source_words = source.split()
    target_words = target.split()

    # Find differences
    diffs = []
    min_len = min(len(source_words), len(target_words))

    for i in range(min_len):
        if source_words[i] != target_words[i]:
            diffs.append((source_words[i], target_words[i]))

    # Handle length differences (insertions/deletions)
    if len(target_words) > len(source_words):
        # Something was inserted - could be missing preposition
        for i in range(len(source_words), len(target_words)):
            if target_words[i] in ['في', 'إلى', 'على', 'من', 'عن', 'ب', 'ل']:
                return 'missing_prep', f"+ {target_words[i]}"

    if not diffs:
        return None, None

    # Analyze first difference
    src, tgt = diffs[0]

    # Gender agreement: masculine ↔ feminine adjective endings
    if src.endswith('ة') and not tgt.endswith('ة'):
        return 'gender_agreement', f"{src} → {tgt}"
    if not src.endswith('ة') and tgt.endswith('ة') and len(src) > 2:
        # Check if it's an adjective getting feminized
        if src + 'ة' == tgt or src[:-1] + 'ة' == tgt:
            return 'gender_agreement', f"{src} → {tgt}"

    # Verb agreement: check for verb conjugation changes
    verb_markers = ['وا', 'ون', 'ين', 'ان', 'ت', 'ن', 'ي']
    if any(src.endswith(m) for m in verb_markers) or any(tgt.endswith(m) for m in verb_markers):
        if src[:-2] == tgt[:-2] or src[:-1] == tgt[:-1]:  # Same root
            return 'verb_agreement', f"{src} → {tgt}"

    # Number agreement: singular ↔ plural
    plural_markers = ['ون', 'ين', 'ات', 'ان']
    if any(tgt.endswith(m) for m in plural_markers) and not any(src.endswith(m) for m in plural_markers):
        return 'number_agreement', f"{src} → {tgt}"
    if any(src.endswith(m) for m in plural_markers) and not any(tgt.endswith(m) for m in plural_markers):
        return 'number_agreement', f"{src} → {tgt}"

    # Wrong preposition
    prepositions = {'في', 'إلى', 'على', 'من', 'عن', 'ب', 'ل', 'الى', 'علي'}
    if src in prepositions and tgt in prepositions:
        # Make sure it's not just alif_maqsura
        if not (src == 'علي' and tgt == 'على') and not (src == 'الي' and tgt == 'إلى'):
            return 'wrong_prep', f"{src} → {tgt}"

    # Definiteness: ال article issues
    if src.startswith('ال') and not tgt.startswith('ال'):
        return 'definiteness', f"{src} → {tgt}"
    if not src.startswith('ال') and tgt.startswith('ال'):
        return 'definiteness', f"{src} → {tgt}"

    # Punctuation
    punct = set('،.؟!؛:,;?')
    if any(c in punct for c in src + tgt):
        return 'punctuation', f"{src} → {tgt}"

    # Spelling (catch-all for other single-word changes)
    if len(diffs) == 1 and len(src) > 2 and len(tgt) > 2:
        return 'spelling', f"{src} → {tgt}"

    return None, None


def extract_grammar_from_qalb():
    """Extract grammar/syntax samples from QALB."""
    print("\n=== Extracting Grammar samples from QALB ===")

    if not QALB_TRAIN.exists():
        print(f"WARNING: QALB file not found: {QALB_TRAIN}")
        return [], []

    qalb_data = load_json(QALB_TRAIN)
    print(f"Loaded {len(qalb_data)} QALB pairs")

    grammar_samples = defaultdict(list)

    for i, pair in enumerate(qalb_data):
        source = pair.get('source', '')
        target = pair.get('target', '')

        # Skip if too short or identical
        if len(source) < 10 or source == target:
            continue

        # Skip if too many differences (want clean single-error examples)
        source_words = source.split()
        target_words = target.split()
        if abs(len(source_words) - len(target_words)) > 2:
            continue

        # Classify the error
        category, correction = classify_qalb_error(source, target)

        if category and category in GRAMMAR_CATEGORIES:
            sample = {
                'id': f"full-{category}-{len(grammar_samples[category]):04d}",
                'source': source,
                'target': target,
                'category': category,
                'correction': correction,
                'source_corpus': 'qalb',
                'difficulty': classify_difficulty(source, target)
            }
            grammar_samples[category].append(sample)

    print(f"\nGrammar samples extracted:")
    for cat in sorted(grammar_samples.keys()):
        print(f"  {cat}: {len(grammar_samples[cat])}")

    # Balance categories (take up to 150 per category)
    balanced = []
    for cat, samples in grammar_samples.items():
        random.shuffle(samples)
        balanced.extend(samples[:150])

    # Split into test/dev (80/20)
    random.shuffle(balanced)
    split_idx = int(len(balanced) * 0.8)

    return balanced[:split_idx], balanced[split_idx:]


def extract_identity_from_corpus(n_samples: int = 500):
    """Extract correct sentences from MSA corpus for false positive testing."""
    print("\n=== Extracting Identity samples from MSA corpus ===")

    if not MSA_CORPUS.exists():
        print(f"WARNING: MSA corpus not found: {MSA_CORPUS}")
        return []

    identity_samples = []
    seen = set()

    # Sample lines from corpus
    with open(MSA_CORPUS, 'r', encoding='utf-8') as f:
        # Read a subset of lines (corpus is huge)
        lines = []
        for i, line in enumerate(f):
            if i % 1000 == 0:  # Sample every 1000th line
                line = line.strip()
                if 10 <= len(line.split()) <= 30:  # Good length
                    lines.append(line)
            if len(lines) >= n_samples * 3:  # Get extra for filtering
                break

    print(f"Sampled {len(lines)} candidate sentences")

    # Filter for clean sentences
    for line in lines:
        # Skip if has obvious issues
        if '?' in line or '!' in line:  # Likely incomplete
            continue
        if line.count('،') > 5:  # Too many commas
            continue
        if len(line) < 20:  # Too short
            continue

        # Deduplicate
        if line in seen:
            continue
        seen.add(line)

        word_count = len(line.split())
        if word_count <= 10:
            difficulty = 'short'
        elif word_count <= 20:
            difficulty = 'medium'
        else:
            difficulty = 'long'

        sample = {
            'id': f"identity-{len(identity_samples):04d}",
            'text': line,
            'word_count': word_count,
            'category': difficulty,
            'source_corpus': 'msa'
        }
        identity_samples.append(sample)

        if len(identity_samples) >= n_samples:
            break

    print(f"Extracted {len(identity_samples)} identity samples")

    # Distribution
    dist = defaultdict(int)
    for s in identity_samples:
        dist[s['category']] += 1
    for cat, count in sorted(dist.items()):
        print(f"  {cat}: {count}")

    return identity_samples


def build_fasih():
    """Build the complete FASIH benchmark suite."""
    print("=" * 60)
    print("BUILDING FASIH BENCHMARK SUITE")
    print("=" * 60)

    # 1. Extract Core (orthographic from v5.1 real samples)
    core_test, core_dev = extract_core_from_v51()

    # 2. Extract Grammar from QALB
    grammar_test, grammar_dev = extract_grammar_from_qalb()

    # 3. Build Full (Core + Grammar)
    full_test = core_test + grammar_test
    full_dev = core_dev + grammar_dev

    # Reassign IDs for full
    for i, sample in enumerate(full_test):
        sample['id'] = f"full-{sample['category']}-{i:04d}"
    for i, sample in enumerate(full_dev):
        sample['id'] = f"full-{sample['category']}-dev-{i:04d}"

    # 4. Extract Identity
    identity_samples = extract_identity_from_corpus(500)

    # 5. Save everything
    print("\n=== Saving FASIH Benchmark ===")

    # Core
    save_json(core_test, FASIH_DIR / "core" / "test.json")
    save_json(core_dev, FASIH_DIR / "core" / "dev.json")

    # Full
    save_json(full_test, FASIH_DIR / "full" / "test.json")
    save_json(full_dev, FASIH_DIR / "full" / "dev.json")

    # Identity
    save_json(identity_samples, FASIH_DIR / "identity" / "test.json")

    # 6. Generate rubric
    rubric = generate_rubric(core_test, full_test, identity_samples)
    save_json(rubric, FASIH_DIR / "rubric.json")

    # 7. Print summary
    print("\n" + "=" * 60)
    print("FASIH BENCHMARK COMPLETE")
    print("=" * 60)
    print(f"\nFASIH-Core:")
    print(f"  Test: {len(core_test)} samples")
    print(f"  Dev:  {len(core_dev)} samples")
    print(f"\nFASIH-Full:")
    print(f"  Test: {len(full_test)} samples")
    print(f"  Dev:  {len(full_dev)} samples")
    print(f"\nFASIH-Identity:")
    print(f"  Test: {len(identity_samples)} samples")
    print(f"\nTotal: {len(core_test) + len(core_dev) + len(grammar_test) + len(grammar_dev) + len(identity_samples)} samples")


def generate_rubric(core: List, full: List, identity: List) -> Dict:
    """Generate the rubric.json with category definitions."""

    # Count by category
    core_dist = defaultdict(int)
    for s in core:
        core_dist[s['category']] += 1

    full_dist = defaultdict(int)
    for s in full:
        full_dist[s['category']] += 1

    rubric = {
        "name": "FASIH Arabic GEC Benchmark",
        "version": "1.0.0",
        "description": "Comprehensive Arabic Grammatical Error Correction benchmark built from real MSA text",
        "source": "https://github.com/your-repo/nahawi",
        "license": "CC-BY-4.0",

        "benchmarks": {
            "core": {
                "description": "Orthographic errors from real MSA Wikipedia text",
                "total_test": len([s for s in core]),
                "categories": list(core_dist.keys()),
                "distribution": dict(core_dist)
            },
            "full": {
                "description": "Complete grammar coverage including morphology and syntax",
                "total_test": len(full),
                "categories": list(full_dist.keys()),
                "distribution": dict(full_dist)
            },
            "identity": {
                "description": "Correct sentences for false positive testing",
                "total_test": len(identity),
                "purpose": "Models should return these unchanged"
            }
        },

        "categories": {
            "hamza": {
                "description": "Incorrect or missing hamza (أ إ آ ء ؤ ئ)",
                "examples": ["اعلان → إعلان", "اكثر → أكثر", "مسؤول → مسؤول"],
                "difficulty": "medium"
            },
            "taa_marbuta": {
                "description": "Confusion between taa marbuta (ة) and haa (ه)",
                "examples": ["المدرسه → المدرسة", "الحكومه → الحكومة"],
                "difficulty": "easy"
            },
            "alif_maqsura": {
                "description": "Confusion between alif maqsura (ى) and yaa (ي)",
                "examples": ["علي → على", "إلي → إلى", "حتي → حتى"],
                "difficulty": "easy"
            },
            "alif_madda": {
                "description": "Missing or incorrect alif madda (آ)",
                "examples": ["الان → الآن", "اخر → آخر", "القران → القرآن"],
                "difficulty": "medium"
            },
            "dad_za": {
                "description": "Confusion between dad (ض) and za (ظ)",
                "examples": ["نضر → نظر", "حضور → حظور", "ضهر → ظهر"],
                "difficulty": "hard"
            },
            "dal_thal": {
                "description": "Confusion between dal (د) and thal (ذ)",
                "examples": ["هدا → هذا", "ادا → إذا", "كدلك → كذلك"],
                "difficulty": "medium"
            },
            "gender_agreement": {
                "description": "Noun-adjective gender mismatch",
                "examples": ["المدينة الكبير → المدينة الكبيرة"],
                "difficulty": "hard"
            },
            "number_agreement": {
                "description": "Noun-adjective or subject-verb number mismatch",
                "examples": ["الطلاب الناجح → الطلاب الناجحون"],
                "difficulty": "hard"
            },
            "verb_agreement": {
                "description": "Subject-verb agreement errors",
                "examples": ["الطالبات درسوا → الطالبات درسن"],
                "difficulty": "hard"
            },
            "missing_prep": {
                "description": "Missing required preposition",
                "examples": ["يبحث المعلومات → يبحث عن المعلومات"],
                "difficulty": "hard"
            },
            "wrong_prep": {
                "description": "Semantically incorrect preposition (NOT alif_maqsura)",
                "examples": ["يعتمد من → يعتمد على", "اشترك بالمؤتمر → اشترك في المؤتمر"],
                "difficulty": "hard"
            },
            "definiteness": {
                "description": "Article agreement errors",
                "examples": ["كتاب الجديد → الكتاب الجديد"],
                "difficulty": "medium"
            },
            "spelling": {
                "description": "Common spelling errors",
                "examples": ["الإقتصاد → الاقتصاد"],
                "difficulty": "medium"
            },
            "punctuation": {
                "description": "Punctuation errors",
                "examples": ["Missing comma, wrong punctuation mark"],
                "difficulty": "medium"
            }
        },

        "evaluation": {
            "primary_metric": "F0.5",
            "rationale": "Precision matters more than recall in GEC - don't introduce new errors",
            "pass_threshold": 0.90,
            "minimum_threshold": 0.80
        }
    }

    return rubric


if __name__ == "__main__":
    random.seed(42)  # Reproducibility
    build_fasih()
