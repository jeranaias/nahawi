#!/usr/bin/env python3
"""
Merge hamza augmentation data with existing LoRA training data.
"""

import json
import random
from collections import Counter

# Paths
EXISTING_TRAIN = "/home/ubuntu/nahawi/data/lora_train_300k.json"
HAMZA_AUG = "/home/ubuntu/nahawi/data/hamza_augmentation.json"
CHAR_AUG = "/home/ubuntu/nahawi/data/single_char_augmentation.json"
OUTPUT = "/home/ubuntu/nahawi/data/lora_train_hamza_aug.json"

def load_json(path):
    print(f"Loading {path}...", flush=True)
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read().strip()

    # Try as JSON array first
    if content.startswith('['):
        data = json.loads(content)
    else:
        # JSONL format (one object per line)
        data = []
        for line in content.split('\n'):
            line = line.strip()
            if line:
                data.append(json.loads(line))

    print(f"  Loaded {len(data):,} pairs", flush=True)
    return data

def main():
    print("=" * 70, flush=True)
    print("MERGING AUGMENTATION DATA", flush=True)
    print("=" * 70, flush=True)

    # Load all datasets
    existing = load_json(EXISTING_TRAIN)
    hamza = load_json(HAMZA_AUG)
    char_aug = load_json(CHAR_AUG)

    print(f"\nDataset sizes:", flush=True)
    print(f"  Existing:  {len(existing):,}", flush=True)
    print(f"  Hamza:     {len(hamza):,}", flush=True)
    print(f"  Char aug:  {len(char_aug):,}", flush=True)

    # Combine with weighting
    # Give hamza 1.5x weight since that's our weakness
    combined = []
    combined.extend(existing)
    combined.extend(hamza)  # All hamza pairs
    combined.extend(hamza[:len(hamza)//2])  # Extra 50% hamza for emphasis
    combined.extend(char_aug)

    print(f"\nCombined: {len(combined):,} pairs", flush=True)

    # Shuffle
    random.seed(42)
    random.shuffle(combined)

    # Deduplicate by source text
    seen_sources = set()
    unique = []
    for item in combined:
        src = item.get('source', item.get('src', ''))
        if src not in seen_sources:
            seen_sources.add(src)
            unique.append(item)

    print(f"After dedup: {len(unique):,} pairs", flush=True)

    # Analyze hamza content
    hamza_chars = 'إأآؤئ'
    with_hamza = sum(1 for item in unique if any(c in item.get('target', item.get('tgt', '')) for c in hamza_chars))
    print(f"Pairs with hamza in target: {with_hamza:,} ({100*with_hamza/len(unique):.1f}%)", flush=True)

    # Save
    print(f"\nSaving to {OUTPUT}...", flush=True)
    with open(OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(unique, f, ensure_ascii=False)

    print(f"\nDone! Created {len(unique):,} training pairs", flush=True)

if __name__ == "__main__":
    main()
