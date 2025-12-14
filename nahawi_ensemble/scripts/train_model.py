#!/usr/bin/env python3
"""
Train individual models in the Nahawi ensemble.

Usage:
    python train_model.py --model general_gec --epochs 5
    python train_model.py --model hamza_fixer --epochs 3 --synthetic-only
    python train_model.py --model space_fixer --epochs 3

Models available:
    - general_gec: General catch-all GEC
    - hamza_fixer: Hamza-specific errors
    - space_fixer: Merge/split errors
    - deleted_word_fixer: Missing words
    - spelling_fixer: Character-level spelling
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import List, Tuple
import random
import json

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nahawi_ensemble.config import config
from nahawi_ensemble.models.arabart_base import (
    GeneralGEC,
    HamzaFixer,
    SpaceFixer,
    DeletedWordFixer,
    SpellingFixer,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def load_tsv_pairs(path: Path, max_pairs: int = None) -> List[Tuple[str, str]]:
    """Load error-correct pairs from TSV file."""
    pairs = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_pairs and i >= max_pairs:
                break
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                pairs.append((parts[0], parts[1]))
    return pairs


def load_synthetic_data(
    synthetic_dir: Path,
    error_type: str = None,
    max_pairs: int = 100000
) -> List[Tuple[str, str]]:
    """Load synthetic training data, optionally filtered by error type."""
    train_path = synthetic_dir / 'train.tsv'

    if not train_path.exists():
        logger.warning(f"Synthetic data not found: {train_path}")
        return []

    pairs = load_tsv_pairs(train_path, max_pairs)
    logger.info(f"Loaded {len(pairs):,} synthetic pairs")

    return pairs


def generate_focused_data(
    base_pairs: List[Tuple[str, str]],
    focus_type: str,
    num_samples: int = 50000
) -> List[Tuple[str, str]]:
    """
    Generate data focused on a specific error type.

    For hamza: pairs with hamza differences
    For space: pairs with different word counts
    etc.
    """
    focused = []

    for src, tgt in base_pairs:
        if focus_type == 'hamza':
            # Has hamza differences
            hamza = set('أإآءؤئا')
            src_hamza = set(c for c in src if c in hamza)
            tgt_hamza = set(c for c in tgt if c in hamza)
            if src_hamza != tgt_hamza or any(c in src for c in hamza):
                focused.append((src, tgt))

        elif focus_type == 'space':
            # Different word counts or merged/split words
            if len(src.split()) != len(tgt.split()):
                focused.append((src, tgt))

        elif focus_type == 'deleted_word':
            # Target has more words than source
            if len(tgt.split()) > len(src.split()):
                focused.append((src, tgt))

        elif focus_type == 'spelling':
            # Character-level differences in words
            src_words = set(src.split())
            tgt_words = set(tgt.split())
            if src_words != tgt_words:
                focused.append((src, tgt))

        else:
            focused.append((src, tgt))

        if len(focused) >= num_samples:
            break

    logger.info(f"Focused data ({focus_type}): {len(focused):,} pairs")
    return focused


def train_model(
    model_name: str,
    epochs: int = 5,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    synthetic_only: bool = False,
    max_train: int = 100000,
    max_val: int = 5000,
):
    """Train a specific model."""

    # Model factory
    model_factories = {
        'general_gec': GeneralGEC,
        'hamza_fixer': HamzaFixer,
        'space_fixer': SpaceFixer,
        'deleted_word_fixer': DeletedWordFixer,
        'spelling_fixer': SpellingFixer,
    }

    focus_types = {
        'general_gec': None,
        'hamza_fixer': 'hamza',
        'space_fixer': 'space',
        'deleted_word_fixer': 'deleted_word',
        'spelling_fixer': 'spelling',
    }

    if model_name not in model_factories:
        raise ValueError(f"Unknown model: {model_name}")

    logger.info("=" * 60)
    logger.info(f"TRAINING: {model_name}")
    logger.info("=" * 60)

    # Load data
    train_pairs = []
    val_pairs = []

    # QALB real data (use only for evaluation if synthetic_only)
    qalb_train = config.data_dir / 'train.tsv'
    qalb_dev = config.data_dir / 'dev.tsv'

    if qalb_train.exists() and not synthetic_only:
        qalb_pairs = load_tsv_pairs(qalb_train)
        logger.info(f"QALB train: {len(qalb_pairs):,} pairs")
        train_pairs.extend(qalb_pairs)

    if qalb_dev.exists():
        val_pairs = load_tsv_pairs(qalb_dev, max_val)
        logger.info(f"QALB dev: {len(val_pairs):,} pairs")

    # Synthetic data
    synthetic_dir = config.base_dir / 'synthetic_v4'
    if synthetic_dir.exists():
        synthetic_pairs = load_synthetic_data(synthetic_dir, max_pairs=max_train)
        train_pairs.extend(synthetic_pairs)

    if not train_pairs:
        logger.error("No training data found!")
        logger.info("Generate synthetic data first with generate_qalb_synthetic_v4.py")
        return

    # Focus data if needed
    focus = focus_types.get(model_name)
    if focus:
        train_pairs = generate_focused_data(train_pairs, focus, max_train)

    # Shuffle
    random.seed(42)
    random.shuffle(train_pairs)

    # Limit size
    if len(train_pairs) > max_train:
        train_pairs = train_pairs[:max_train]

    logger.info(f"Final training size: {len(train_pairs):,}")
    logger.info(f"Validation size: {len(val_pairs):,}")

    # Create model
    model = model_factories[model_name](device=config.device)
    model.load()

    # Output directory
    output_dir = config.models_dir / model_name
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

    logger.info(f"Training complete! Model saved to: {output_dir}")

    # Save training info
    info = {
        'model': model_name,
        'epochs': epochs,
        'train_size': len(train_pairs),
        'val_size': len(val_pairs),
        'focus_type': focus,
        'synthetic_only': synthetic_only,
    }
    with open(output_dir / 'training_info.json', 'w') as f:
        json.dump(info, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Train Nahawi ensemble models')
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['general_gec', 'hamza_fixer', 'space_fixer', 'deleted_word_fixer', 'spelling_fixer'],
        help='Which model to train'
    )
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--synthetic-only', action='store_true', help='Use only synthetic data')
    parser.add_argument('--max-train', type=int, default=100000, help='Max training examples')
    parser.add_argument('--max-val', type=int, default=5000, help='Max validation examples')
    args = parser.parse_args()

    train_model(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        synthetic_only=args.synthetic_only,
        max_train=args.max_train,
        max_val=args.max_val,
    )


if __name__ == '__main__':
    main()
