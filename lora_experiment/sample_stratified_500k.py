#!/usr/bin/env python3
"""
Create stratified 500K synthetic dataset for LoRA experiment.
Samples diversely across error types and difficulty levels.
"""

import json
import random
from collections import defaultdict
from pathlib import Path

# Paths
QALB_PATH = "/home/ubuntu/nahawi/data/qalb_real_train.json"
ULTIMATE_PATH = "/home/ubuntu/nahawi/data/ultimate"
OUTPUT_PATH = "/home/ubuntu/nahawi/data/lora_train_500k.json"

# Target: 37K QALB + 463K synthetic = 500K total
QALB_COUNT = 37000  # All real data
SYNTHETIC_COUNT = 463000

# Error type detection heuristics
def detect_error_types(source, target):
    """Detect what error types are present in this pair."""
    errors = set()

    # Hamza errors (أ/ا/إ/آ)
    hamza_chars = 'أإآءؤئ'
    alif = 'ا'
    for c in hamza_chars:
        if c in target and c not in source:
            errors.add('hamza')
            break
        if c in source and alif in target:
            errors.add('hamza')
            break

    # Taa marbuta (ة/ه)
    if 'ة' in target and 'ه' in source:
        errors.add('taa_marbuta')
    if 'ه' in target and 'ة' in source:
        errors.add('taa_marbuta')

    # Alif maqsura (ى/ي)
    if 'ى' in target and 'ي' in source:
        errors.add('alif_maqsura')
    if 'ي' in target and 'ى' in source:
        errors.add('alif_maqsura')

    # Space errors
    src_words = source.split()
    tgt_words = target.split()
    if len(src_words) != len(tgt_words):
        errors.add('spacing')

    # Punctuation
    punct = '،؛؟!.,:;?'
    src_punct = sum(1 for c in source if c in punct)
    tgt_punct = sum(1 for c in target if c in punct)
    if src_punct != tgt_punct:
        errors.add('punctuation')

    # Letter confusion (ض/ظ, د/ذ, ت/ط, س/ص)
    confusion_pairs = [('ض', 'ظ'), ('د', 'ذ'), ('ت', 'ط'), ('س', 'ص')]
    for c1, c2 in confusion_pairs:
        if (c1 in source and c2 in target) or (c2 in source and c1 in target):
            errors.add('letter_confusion')
            break

    # Length-based difficulty
    edit_ratio = abs(len(target) - len(source)) / max(len(source), 1)
    if edit_ratio < 0.1:
        errors.add('easy')
    elif edit_ratio < 0.25:
        errors.add('medium')
    else:
        errors.add('hard')

    if not errors:
        errors.add('other')

    return errors


def load_jsonl(path):
    """Load JSONL file."""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                src = item.get('source', item.get('src', ''))
                tgt = item.get('target', item.get('tgt', ''))
                if src and tgt:
                    data.append((src, tgt))
            except:
                pass
    return data


def load_json_array(path):
    """Load JSON array file."""
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        if content.startswith('['):
            items = json.loads(content)
        else:
            # JSONL fallback
            items = [json.loads(line) for line in content.split('\n') if line.strip()]

    data = []
    for item in items:
        src = item.get('source', item.get('src', ''))
        tgt = item.get('target', item.get('tgt', ''))
        if src and tgt:
            data.append((src, tgt))
    return data


def main():
    print("=" * 60)
    print("STRATIFIED 500K SAMPLING FOR LORA EXPERIMENT")
    print("=" * 60)

    # Load QALB real data
    print("\nLoading QALB real data...")
    try:
        qalb = load_json_array(QALB_PATH)
    except:
        qalb = load_jsonl(QALB_PATH)
    print(f"  Loaded {len(qalb):,} QALB pairs")

    # Load synthetic data from ultimate buckets
    print("\nLoading synthetic data...")
    synthetic_by_difficulty = {
        'easy': [],
        'medium': [],
        'hard': []
    }

    for difficulty in ['easy', 'medium', 'hard']:
        path = f"{ULTIMATE_PATH}/train_{difficulty}.json"
        try:
            data = load_jsonl(path)
            synthetic_by_difficulty[difficulty] = data
            print(f"  {difficulty}: {len(data):,} pairs")
        except Exception as e:
            print(f"  {difficulty}: Error loading - {e}")

    # Classify synthetic by error type
    print("\nClassifying synthetic data by error type...")
    synthetic_by_type = defaultdict(list)

    all_synthetic = []
    for diff, pairs in synthetic_by_difficulty.items():
        all_synthetic.extend(pairs)

    # Sample for classification (too slow to classify all)
    sample_size = min(1000000, len(all_synthetic))
    sample = random.sample(all_synthetic, sample_size) if len(all_synthetic) > sample_size else all_synthetic

    for src, tgt in sample:
        error_types = detect_error_types(src, tgt)
        for et in error_types:
            synthetic_by_type[et].append((src, tgt))

    print("  Error type distribution:")
    for et, pairs in sorted(synthetic_by_type.items(), key=lambda x: -len(x[1])):
        print(f"    {et}: {len(pairs):,}")

    # Stratified sampling: equal representation of each error type
    print(f"\nStratified sampling {SYNTHETIC_COUNT:,} synthetic pairs...")

    error_types = list(synthetic_by_type.keys())
    per_type = SYNTHETIC_COUNT // len(error_types)

    sampled_synthetic = []
    seen_sources = set()  # Dedup

    for et in error_types:
        available = synthetic_by_type[et]
        count = min(per_type, len(available))

        # Sample without replacement, dedup by source
        sampled = 0
        for src, tgt in random.sample(available, min(len(available), count * 2)):
            if src not in seen_sources:
                seen_sources.add(src)
                sampled_synthetic.append((src, tgt))
                sampled += 1
                if sampled >= count:
                    break

        print(f"    {et}: sampled {sampled:,}")

    # Fill remaining with random samples
    remaining = SYNTHETIC_COUNT - len(sampled_synthetic)
    if remaining > 0:
        print(f"  Filling {remaining:,} remaining slots with random samples...")
        random.shuffle(all_synthetic)
        for src, tgt in all_synthetic:
            if src not in seen_sources:
                seen_sources.add(src)
                sampled_synthetic.append((src, tgt))
                if len(sampled_synthetic) >= SYNTHETIC_COUNT:
                    break

    print(f"  Total synthetic sampled: {len(sampled_synthetic):,}")

    # Combine QALB + synthetic
    print("\nCombining QALB + synthetic...")
    final_data = []

    # Add all QALB
    for src, tgt in qalb:
        final_data.append({'source': src, 'target': tgt, 'type': 'qalb'})

    # Add sampled synthetic
    for src, tgt in sampled_synthetic:
        final_data.append({'source': src, 'target': tgt, 'type': 'synthetic'})

    # Shuffle
    random.shuffle(final_data)

    print(f"  Final dataset: {len(final_data):,} pairs")
    print(f"    QALB: {len(qalb):,} ({100*len(qalb)/len(final_data):.1f}%)")
    print(f"    Synthetic: {len(sampled_synthetic):,} ({100*len(sampled_synthetic)/len(final_data):.1f}%)")

    # Save
    print(f"\nSaving to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for item in final_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print("\nDone!")
    print("=" * 60)


if __name__ == "__main__":
    main()
