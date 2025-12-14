#!/usr/bin/env python3
"""
Train all AraBART-based models in the ensemble.

This script:
1. Generates synthetic data if needed
2. Trains all models sequentially (or can be run in parallel instances)

Usage:
    python train_all.py                    # Train all models
    python train_all.py --model general_gec  # Train specific model
    python train_all.py --generate-only    # Only generate data
"""

import sys
import os
import argparse
import logging
import json
import random
from pathlib import Path
from typing import List, Tuple, Dict
from datetime import datetime

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
NAHAWI_DIR = Path('C:/nahawi')
QALB_DATA_DIR = NAHAWI_DIR / 'qalb_real_data'
WIKI_SENTENCES = NAHAWI_DIR / 'arabic_wiki' / 'sentences.txt'
QALB_PATTERNS = NAHAWI_DIR / 'qalb_correct_to_errors.json'
SYNTHETIC_DIR = NAHAWI_DIR / 'synthetic_v4'
CHECKPOINTS_DIR = NAHAWI_DIR / 'nahawi_ensemble' / 'checkpoints'


def load_tsv_pairs(path: Path, max_pairs: int = None) -> List[Tuple[str, str]]:
    """Load pairs from TSV."""
    pairs = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_pairs and i >= max_pairs:
                break
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                pairs.append((parts[0], parts[1]))
    return pairs


def generate_synthetic_data(num_samples: int = 200000):
    """Generate synthetic training data using v4 generator."""
    logger.info("=" * 60)
    logger.info("GENERATING SYNTHETIC DATA")
    logger.info("=" * 60)

    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)

    # Check if already exists
    train_path = SYNTHETIC_DIR / 'train.tsv'
    if train_path.exists():
        existing = sum(1 for _ in open(train_path, 'r', encoding='utf-8'))
        if existing >= num_samples * 0.9:
            logger.info(f"Synthetic data already exists: {existing:,} pairs")
            return True

    # Import generator
    try:
        sys.path.insert(0, str(NAHAWI_DIR))
        from generate_qalb_synthetic_v4 import MorphologyAwareGenerator, load_sentences
    except ImportError as e:
        logger.error(f"Cannot import generator: {e}")
        return False

    # Load source sentences
    sources = []
    if WIKI_SENTENCES.exists():
        sources.append(WIKI_SENTENCES)
    if (QALB_DATA_DIR / 'train.tsv').exists():
        sources.append(QALB_DATA_DIR / 'train.tsv')

    if not sources:
        logger.error("No source data found!")
        return False

    sentences = load_sentences(sources, max_sentences=300000)
    logger.info(f"Loaded {len(sentences):,} source sentences")

    # Initialize generator
    generator = MorphologyAwareGenerator(
        qalb_patterns_path=QALB_PATTERNS,
        error_rate=0.85,
        clean_rate=0.15,
        seed=42
    )

    # Generate pairs
    logger.info(f"Generating {num_samples:,} pairs...")
    rng = random.Random(42)
    seen = set()

    with open(train_path, 'w', encoding='utf-8') as f:
        i = 0
        attempts = 0
        max_attempts = num_samples * 3

        while i < num_samples and attempts < max_attempts:
            attempts += 1
            sentence = rng.choice(sentences)
            error_text, correct_text, _ = generator.generate_error(sentence)

            pair = (error_text, correct_text)
            if pair not in seen:
                seen.add(pair)
                f.write(f"{error_text}\t{correct_text}\n")
                i += 1

                if i % 50000 == 0:
                    logger.info(f"  Generated {i:,} / {num_samples:,}")

    logger.info(f"Generated {i:,} pairs")
    generator.print_stats()

    # Generate dev set
    dev_path = SYNTHETIC_DIR / 'dev.tsv'
    dev_size = min(5000, num_samples // 20)
    logger.info(f"Generating {dev_size:,} dev pairs...")

    with open(dev_path, 'w', encoding='utf-8') as f:
        for j in range(dev_size):
            sentence = rng.choice(sentences)
            error_text, correct_text, _ = generator.generate_error(sentence)
            f.write(f"{error_text}\t{correct_text}\n")

    logger.info(f"Synthetic data saved to: {SYNTHETIC_DIR}")
    return True


def prepare_focused_data(focus_type: str, max_samples: int = 100000) -> List[Tuple[str, str]]:
    """Prepare data focused on a specific error type."""

    # Load all available data
    all_pairs = []

    # QALB real data
    qalb_train = QALB_DATA_DIR / 'train.tsv'
    if qalb_train.exists():
        all_pairs.extend(load_tsv_pairs(qalb_train))

    # Synthetic data
    synthetic_train = SYNTHETIC_DIR / 'train.tsv'
    if synthetic_train.exists():
        all_pairs.extend(load_tsv_pairs(synthetic_train, max_pairs=200000))

    logger.info(f"Total pairs for filtering: {len(all_pairs):,}")

    # Filter based on focus type
    focused = []

    for src, tgt in all_pairs:
        include = False

        if focus_type == 'hamza':
            hamza = set('أإآءؤئا')
            if any(c in src for c in hamza) or any(c in tgt for c in hamza):
                # Check if there's a hamza difference
                src_h = ''.join(c for c in src if c in hamza)
                tgt_h = ''.join(c for c in tgt if c in hamza)
                if src_h != tgt_h:
                    include = True

        elif focus_type == 'space':
            if len(src.split()) != len(tgt.split()):
                include = True

        elif focus_type == 'deleted_word':
            if len(tgt.split()) > len(src.split()):
                include = True

        elif focus_type == 'spelling':
            # Any word-level differences
            if set(src.split()) != set(tgt.split()):
                include = True

        elif focus_type == 'general':
            include = True

        if include:
            focused.append((src, tgt))

        if len(focused) >= max_samples:
            break

    logger.info(f"Focused data ({focus_type}): {len(focused):,} pairs")
    return focused


def train_model(
    model_name: str,
    epochs: int = 5,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    max_train: int = 100000,
):
    """Train a single model."""
    from nahawi_ensemble.models.arabart_base import (
        GeneralGEC, HamzaFixer, SpaceFixer, DeletedWordFixer, SpellingFixer
    )
    from nahawi_ensemble.config import config

    model_factories = {
        'general_gec': (GeneralGEC, 'general'),
        'hamza_fixer': (HamzaFixer, 'hamza'),
        'space_fixer': (SpaceFixer, 'space'),
        'deleted_word_fixer': (DeletedWordFixer, 'deleted_word'),
        'spelling_fixer': (SpellingFixer, 'spelling'),
    }

    if model_name not in model_factories:
        logger.error(f"Unknown model: {model_name}")
        return False

    factory, focus_type = model_factories[model_name]

    logger.info("=" * 60)
    logger.info(f"TRAINING: {model_name}")
    logger.info(f"Focus: {focus_type}")
    logger.info("=" * 60)

    # Prepare data
    train_pairs = prepare_focused_data(focus_type, max_train)

    if not train_pairs:
        logger.error("No training data!")
        return False

    # Validation data
    val_pairs = []
    qalb_dev = QALB_DATA_DIR / 'dev.tsv'
    if qalb_dev.exists():
        val_pairs = load_tsv_pairs(qalb_dev, max_pairs=2000)

    # Shuffle
    random.shuffle(train_pairs)

    logger.info(f"Training: {len(train_pairs):,} pairs")
    logger.info(f"Validation: {len(val_pairs):,} pairs")

    # Create model
    model = factory(device=config.device)
    model.load()

    # Output dir
    output_dir = CHECKPOINTS_DIR / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Train
    model.train(
        train_pairs=train_pairs,
        val_pairs=val_pairs if val_pairs else None,
        output_dir=str(output_dir),
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        fp16=True,
    )

    # Save info
    info = {
        'model': model_name,
        'focus_type': focus_type,
        'epochs': epochs,
        'train_size': len(train_pairs),
        'val_size': len(val_pairs),
        'timestamp': datetime.now().isoformat(),
    }
    with open(output_dir / 'training_info.json', 'w') as f:
        json.dump(info, f, indent=2)

    logger.info(f"Model saved to: {output_dir}")
    return True


def main():
    parser = argparse.ArgumentParser(description='Train all Nahawi ensemble models')
    parser.add_argument('--model', type=str, default=None,
                        choices=['general_gec', 'hamza_fixer', 'space_fixer',
                                 'deleted_word_fixer', 'spelling_fixer'],
                        help='Train specific model (default: all)')
    parser.add_argument('--generate-only', action='store_true',
                        help='Only generate synthetic data')
    parser.add_argument('--skip-generate', action='store_true',
                        help='Skip data generation')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-train', type=int, default=100000)
    parser.add_argument('--synthetic-samples', type=int, default=200000)
    args = parser.parse_args()

    start_time = datetime.now()
    logger.info(f"Starting at {start_time}")

    # Step 1: Generate synthetic data
    if not args.skip_generate:
        if not generate_synthetic_data(args.synthetic_samples):
            logger.error("Failed to generate synthetic data!")
            return

    if args.generate_only:
        logger.info("Data generation complete. Exiting.")
        return

    # Step 2: Train models
    models_to_train = [args.model] if args.model else [
        'general_gec',
        'hamza_fixer',
        'space_fixer',
        'deleted_word_fixer',
        'spelling_fixer',
    ]

    for model_name in models_to_train:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training {model_name}...")
        logger.info(f"{'='*60}\n")

        success = train_model(
            model_name=model_name,
            epochs=args.epochs,
            batch_size=args.batch_size,
            max_train=args.max_train,
        )

        if not success:
            logger.error(f"Failed to train {model_name}")

    end_time = datetime.now()
    duration = end_time - start_time
    logger.info(f"\nTotal time: {duration}")
    logger.info("All training complete!")


if __name__ == '__main__':
    main()
