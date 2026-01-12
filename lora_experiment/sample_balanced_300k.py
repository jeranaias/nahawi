#!/usr/bin/env python3
"""
Create balanced training dataset for LoRA experiment.
Per expert recommendation: 1:4 ratio, repeat QALB 4x.

Target: 150K QALB (repeated) + 150K synthetic = 300K total
"""

import json
import random
from collections import defaultdict
from pathlib import Path

# Paths
QALB_PATH = "/home/ubuntu/nahawi/data/qalb_real_train.json"
ULTIMATE_PATH = "/home/ubuntu/nahawi/data/ultimate"
OUTPUT_PATH = "/home/ubuntu/nahawi/data/lora_train_300k.json"

# Target sizes (1:4 ratio = repeat QALB 4x to match synthetic)
QALB_REPEATS = 4  # Repeat QALB 4x
SYNTHETIC_COUNT = 150000  # Match effective QALB size


def detect_error_types(source, target):
    """Detect what error types are present in this pair."""
    errors = set()

    # Hamza errors
    hamza_chars = 'أإآءؤئ'
    if any(c in target and c not in source for c in hamza_chars):
        errors.add('hamza')
    if any(c in source and c not in target for c in hamza_chars):
        errors.add('hamza')

    # Taa marbuta
    if ('ة' in target and 'ه' in source) or ('ه' in target and 'ة' in source):
        errors.add('taa_marbuta')

    # Alif maqsura
    if ('ى' in target and 'ي' in source) or ('ي' in target and 'ى' in source):
        errors.add('alif_maqsura')

    # Space errors
    if len(source.split()) != len(target.split()):
        errors.add('spacing')

    # Punctuation
    punct = '،؛؟!.,:;?'
    if sum(1 for c in source if c in punct) != sum(1 for c in target if c in punct):
        errors.add('punctuation')

    # Letter confusion
    confusion_pairs = [('ض', 'ظ'), ('د', 'ذ'), ('ت', 'ط'), ('س', 'ص')]
    for c1, c2 in confusion_pairs:
        if (c1 in source and c2 in target) or (c2 in source and c1 in target):
            errors.add('letter_confusion')
            break

    # Difficulty based on edit ratio
    edit_ratio = abs(len(target) - len(source)) / max(len(source), 1)
    if edit_ratio >= 0.25:
        errors.add('hard')
    elif edit_ratio >= 0.1:
        errors.add('medium')

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
    print("BALANCED 300K SAMPLING (1:4 RATIO)")
    print("=" * 60)
    print("Strategy: Repeat QALB 4x + stratified synthetic")
    print()

    # Load QALB real data
    print("Loading QALB real data...")
    try:
        qalb = load_json_array(QALB_PATH)
    except:
        qalb = load_jsonl(QALB_PATH)
    print(f"  Loaded {len(qalb):,} QALB pairs")

    # Repeat QALB 4x
    qalb_repeated = qalb * QALB_REPEATS
    print(f"  After 4x repeat: {len(qalb_repeated):,} pairs")

    # Load synthetic data
    print("\nLoading synthetic data from ultimate buckets...")
    all_synthetic = []

    for difficulty in ['easy', 'medium', 'hard']:
        path = f"{ULTIMATE_PATH}/train_{difficulty}.json"
        try:
            data = load_jsonl(path)
            all_synthetic.extend(data)
            print(f"  {difficulty}: {len(data):,} pairs")
        except Exception as e:
            print(f"  {difficulty}: Error - {e}")

    print(f"  Total synthetic available: {len(all_synthetic):,}")

    # Classify synthetic by error type
    print("\nClassifying synthetic by error type...")
    synthetic_by_type = defaultdict(list)

    # Sample for classification (faster)
    sample_size = min(500000, len(all_synthetic))
    sample = random.sample(all_synthetic, sample_size)

    for src, tgt in sample:
        error_types = detect_error_types(src, tgt)
        for et in error_types:
            synthetic_by_type[et].append((src, tgt))

    print("  Distribution:")
    for et, pairs in sorted(synthetic_by_type.items(), key=lambda x: -len(x[1])):
        print(f"    {et}: {len(pairs):,}")

    # Stratified sampling
    print(f"\nStratified sampling {SYNTHETIC_COUNT:,} synthetic pairs...")
    error_types = list(synthetic_by_type.keys())
    per_type = SYNTHETIC_COUNT // len(error_types)

    sampled_synthetic = []
    seen_sources = set()

    for et in error_types:
        available = synthetic_by_type[et]
        count = min(per_type, len(available))

        sampled = 0
        for src, tgt in random.sample(available, min(len(available), count * 2)):
            if src not in seen_sources:
                seen_sources.add(src)
                sampled_synthetic.append((src, tgt))
                sampled += 1
                if sampled >= count:
                    break

        print(f"    {et}: {sampled:,}")

    # Fill remaining
    remaining = SYNTHETIC_COUNT - len(sampled_synthetic)
    if remaining > 0:
        print(f"  Filling {remaining:,} remaining...")
        random.shuffle(all_synthetic)
        for src, tgt in all_synthetic:
            if src not in seen_sources:
                seen_sources.add(src)
                sampled_synthetic.append((src, tgt))
                if len(sampled_synthetic) >= SYNTHETIC_COUNT:
                    break

    print(f"  Total synthetic: {len(sampled_synthetic):,}")

    # Combine
    print("\nCombining datasets...")
    final_data = []

    for src, tgt in qalb_repeated:
        final_data.append({'source': src, 'target': tgt, 'type': 'qalb'})

    for src, tgt in sampled_synthetic:
        final_data.append({'source': src, 'target': tgt, 'type': 'synthetic'})

    random.shuffle(final_data)

    qalb_count = len(qalb_repeated)
    synth_count = len(sampled_synthetic)
    total = len(final_data)

    print(f"\nFinal dataset: {total:,} pairs")
    print(f"  QALB (4x repeated): {qalb_count:,} ({100*qalb_count/total:.1f}%)")
    print(f"  Synthetic: {synth_count:,} ({100*synth_count/total:.1f}%)")
    print(f"  Effective ratio: 1:{synth_count//(qalb_count//QALB_REPEATS):.1f}")

    # Save
    print(f"\nSaving to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for item in final_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print("\nDone!")
    print("=" * 60)


if __name__ == "__main__":
    main()
