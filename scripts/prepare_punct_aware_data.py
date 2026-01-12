#!/usr/bin/env python3
"""
Prepare training data for punct-aware LoRA:
- QALB x50 (WITH punct - teaches punct placement)
- Synthetic 150K (punct STRIPPED - teaches content only)
- Hamza aug (punct STRIPPED - teaches content only)

Principle: Model learns punct ONLY from QALB, content from all sources.
"""

import json
import random
from pathlib import Path

# Punctuation characters to strip
PUNCT_SET = set('،.؟!؛:,;?')

# Paths
QALB_TRAIN = "/home/ubuntu/nahawi/data/qalb_real_train.json"
SYNTHETIC = "/home/ubuntu/nahawi/data/ultimate/train_full.json"
HAMZA_AUG = "/home/ubuntu/nahawi/data/hamza_augmentation.json"
OUTPUT_DIR = "/home/ubuntu/nahawi/data/punct_aware/"


def strip_punct(text):
    """Remove all punctuation tokens and characters from text."""
    # First, remove all punct characters from the string
    text_no_punct = ''.join(c for c in text if c not in PUNCT_SET)

    # Then clean up whitespace
    tokens = text_no_punct.split()
    result = [tok for tok in tokens if tok]  # Remove empty tokens

    return ' '.join(result)


def load_json_streaming(path, limit=None):
    """Load JSON or JSONL file, optionally limiting to first N items."""
    print(f"Loading {path}...")
    data = []

    with open(path, 'r', encoding='utf-8') as f:
        # Try to detect format
        first_char = f.read(1)
        f.seek(0)

        if first_char == '[':
            # Standard JSON array
            data = json.load(f)
        else:
            # JSONL format
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
                    if limit and len(data) >= limit:
                        break

    if limit and len(data) > limit:
        data = data[:limit]

    print(f"  Loaded {len(data)} items")
    return data


def main():
    print("=" * 70)
    print("PREPARE PUNCT-AWARE TRAINING DATA")
    print("=" * 70)
    print("\nStrategy:")
    print("  - QALB x50 WITH punct -> teaches punct placement")
    print("  - Synthetic 150K punct STRIPPED -> teaches content only")
    print("  - Hamza aug punct STRIPPED -> teaches content only")
    print()

    # Create output directory
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # 1. Load QALB and repeat 50x (WITH punct)
    print("\n" + "=" * 50)
    print("Step 1: QALB x50 (WITH punct)")
    print("=" * 50)

    qalb = load_json_streaming(QALB_TRAIN)
    qalb_repeated = []
    for _ in range(50):
        qalb_repeated.extend(qalb)
    random.shuffle(qalb_repeated)

    print(f"  QALB original: {len(qalb)}")
    print(f"  QALB x50: {len(qalb_repeated)}")

    # Sample punct stats
    punct_count = sum(1 for item in qalb[:100]
                      for c in item.get('target', item.get('tgt', ''))
                      if c in PUNCT_SET)
    print(f"  Punct in first 100 targets: {punct_count}")

    # 2. Load synthetic and STRIP punct
    print("\n" + "=" * 50)
    print("Step 2: Synthetic 150K (punct STRIPPED)")
    print("=" * 50)

    # Sample 150K from synthetic
    synthetic_raw = load_json_streaming(SYNTHETIC, limit=200000)  # Load extra for filtering
    random.shuffle(synthetic_raw)

    synthetic_stripped = []
    for item in synthetic_raw:
        src = item.get('source', item.get('src', ''))
        tgt = item.get('target', item.get('tgt', ''))

        # Strip punct from both
        src_clean = strip_punct(src)
        tgt_clean = strip_punct(tgt)

        # Skip if too short after stripping
        if len(src_clean.split()) < 3 or len(tgt_clean.split()) < 3:
            continue

        synthetic_stripped.append({
            'source': src_clean,
            'target': tgt_clean
        })

        if len(synthetic_stripped) >= 150000:
            break

    print(f"  Synthetic stripped: {len(synthetic_stripped)}")

    # Verify no punct
    punct_check = sum(1 for item in synthetic_stripped[:100]
                      for c in item['target'] if c in PUNCT_SET)
    print(f"  Punct in first 100 stripped targets: {punct_check} (should be 0)")

    # 3. Load hamza augmentation and STRIP punct
    print("\n" + "=" * 50)
    print("Step 3: Hamza augmentation (punct STRIPPED)")
    print("=" * 50)

    try:
        hamza_raw = load_json_streaming(HAMZA_AUG)

        hamza_stripped = []
        for item in hamza_raw:
            src = item.get('source', item.get('src', ''))
            tgt = item.get('target', item.get('tgt', ''))

            src_clean = strip_punct(src)
            tgt_clean = strip_punct(tgt)

            if len(src_clean.split()) < 3 or len(tgt_clean.split()) < 3:
                continue

            hamza_stripped.append({
                'source': src_clean,
                'target': tgt_clean
            })

        print(f"  Hamza stripped: {len(hamza_stripped)}")
    except FileNotFoundError:
        print("  Hamza augmentation not found, skipping")
        hamza_stripped = []

    # 4. Combine all data
    print("\n" + "=" * 50)
    print("Step 4: Combine datasets")
    print("=" * 50)

    combined = []

    # Add QALB (WITH punct)
    for item in qalb_repeated:
        combined.append({
            'source': item.get('source', item.get('src', '')),
            'target': item.get('target', item.get('tgt', '')),
            'type': 'qalb'
        })

    # Add synthetic (punct stripped)
    for item in synthetic_stripped:
        combined.append({
            'source': item['source'],
            'target': item['target'],
            'type': 'synthetic_nopunct'
        })

    # Add hamza (punct stripped)
    for item in hamza_stripped:
        combined.append({
            'source': item['source'],
            'target': item['target'],
            'type': 'hamza_nopunct'
        })

    random.shuffle(combined)

    print(f"\nFinal dataset composition:")
    print(f"  QALB x50 (with punct):     {len(qalb_repeated):,}")
    print(f"  Synthetic (no punct):      {len(synthetic_stripped):,}")
    print(f"  Hamza aug (no punct):      {len(hamza_stripped):,}")
    print(f"  TOTAL:                     {len(combined):,}")

    # Calculate ratios
    qalb_ratio = len(qalb_repeated) / len(combined) * 100
    print(f"\n  QALB ratio: {qalb_ratio:.1f}% (punct signal)")

    # 5. Save
    print("\n" + "=" * 50)
    print("Step 5: Save datasets")
    print("=" * 50)

    output_path = Path(OUTPUT_DIR) / "train_punct_aware.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {output_path}")
    print(f"  Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Save type counts for verification
    type_counts = {}
    for item in combined:
        t = item.get('type', 'unknown')
        type_counts[t] = type_counts.get(t, 0) + 1

    stats_path = Path(OUTPUT_DIR) / "stats.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump({
            'total': len(combined),
            'by_type': type_counts,
            'qalb_ratio': qalb_ratio
        }, f, indent=2)
    print(f"  Stats: {stats_path}")

    # 6. Show samples
    print("\n" + "=" * 50)
    print("SAMPLES")
    print("=" * 50)

    print("\nQALB sample (WITH punct):")
    for item in combined:
        if item['type'] == 'qalb':
            print(f"  SRC: {item['source'][:80]}...")
            print(f"  TGT: {item['target'][:80]}...")
            break

    print("\nSynthetic sample (NO punct):")
    for item in combined:
        if item['type'] == 'synthetic_nopunct':
            print(f"  SRC: {item['source'][:80]}...")
            print(f"  TGT: {item['target'][:80]}...")
            break

    print("\n" + "=" * 70)
    print("DONE - Ready for training")
    print("=" * 70)


if __name__ == "__main__":
    main()
