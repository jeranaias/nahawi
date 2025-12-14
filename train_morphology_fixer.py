#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train MorphologyFixer using CAMeLBERT for morphological error correction.

This model handles:
- Gender agreement (masculine/feminine)
- Number agreement (singular/dual/plural)
- Definiteness errors
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import os
import json
import random
import argparse
from pathlib import Path
from typing import List, Tuple, Optional
from datetime import datetime
import logging
import re

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    EncoderDecoderModel,
    BertConfig,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)

# Paths
NAHAWI_DIR = Path('C:/nahawi')
QALB_DIR = NAHAWI_DIR / 'qalb_real_data'
SYNTHETIC_DIR = NAHAWI_DIR / 'synthetic_v4'
OUTPUT_DIR = NAHAWI_DIR / 'nahawi_ensemble' / 'checkpoints'

# Morphology error indicators
MORPHOLOGY_INDICATORS = {
    # Gender markers
    'feminine_end': 'ة',
    'alif_maksura': 'ى',
    # Definiteness
    'definite': 'ال',
    # Common morphological suffixes
    'dual_m': 'ان',
    'dual_f': 'تان',
    'plural_m': 'ون',
    'plural_f': 'ات',
}


class GECDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str]], tokenizer, max_length: int = 128):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        source, target = self.pairs[idx]

        source_enc = self.tokenizer(
            source,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        target_enc = self.tokenizer(
            target,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'input_ids': source_enc['input_ids'].squeeze(),
            'attention_mask': source_enc['attention_mask'].squeeze(),
            'labels': target_enc['input_ids'].squeeze(),
        }


def has_morphology_diff(source: str, target: str) -> bool:
    """
    Check if difference involves morphological changes.

    Morphological changes include:
    - Gender suffix changes (e.g., adding/removing ة)
    - Number suffix changes (ون, ات, ان, etc.)
    - Definiteness changes (adding/removing ال)
    """
    if source == target:
        return False

    source_words = source.split()
    target_words = target.split()

    # Must have same number of words (morphology, not deletion)
    if len(source_words) != len(target_words):
        return False

    for sw, tw in zip(source_words, target_words):
        if sw == tw:
            continue

        # Check for suffix changes
        # Gender: adding/removing ة
        if sw + 'ة' == tw or sw == tw + 'ة':
            return True
        if sw.rstrip('ة') == tw.rstrip('ة') and (sw.endswith('ة') != tw.endswith('ة')):
            return True

        # Definiteness: adding/removing ال
        if 'ال' + sw == tw or sw == 'ال' + tw:
            return True
        if sw.lstrip('ال') == tw.lstrip('ال') and (sw.startswith('ال') != tw.startswith('ال')):
            return True

        # Number suffixes
        number_suffixes = ['ون', 'ين', 'ات', 'ان', 'تان', 'تين']
        for suf in number_suffixes:
            if sw.endswith(suf) != tw.endswith(suf):
                base_s = sw.rstrip(suf) if sw.endswith(suf) else sw
                base_t = tw.rstrip(suf) if tw.endswith(suf) else tw
                # Check if it's roughly the same word
                if len(base_s) > 2 and len(base_t) > 2:
                    if base_s[:3] == base_t[:3]:
                        return True

    return False


def filter_morphology_pairs(pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Filter pairs for morphological errors."""
    filtered = []
    for source, target in pairs:
        if has_morphology_diff(source, target):
            filtered.append((source, target))
    return filtered


def load_pairs(path: Path, max_pairs: int = None) -> List[Tuple[str, str]]:
    pairs = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_pairs and i >= max_pairs:
                break
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                pairs.append((parts[0], parts[1]))
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--max-train', type=int, default=30000)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--use-arabart', action='store_true',
                        help='Use AraBART instead of CAMeLBERT (CAMeLBERT is encoder-only)')
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("TRAINING: MorphologyFixer")
    logger.info("=" * 60)

    # Load all data
    all_pairs = []

    qalb_train = QALB_DIR / 'train.tsv'
    if qalb_train.exists():
        pairs = load_pairs(qalb_train)
        logger.info(f"QALB train: {len(pairs)} pairs")
        all_pairs.extend(pairs)

    synthetic_train = SYNTHETIC_DIR / 'train.tsv'
    if synthetic_train.exists():
        pairs = load_pairs(synthetic_train, args.max_train * 2)
        logger.info(f"Synthetic: {len(pairs)} pairs")
        all_pairs.extend(pairs)

    if not all_pairs:
        logger.error("No training data!")
        return

    # Filter for morphology errors
    logger.info("Filtering for morphology errors...")
    train_pairs = filter_morphology_pairs(all_pairs)
    logger.info(f"Filtered: {len(train_pairs)} morphology pairs")

    if len(train_pairs) < 100:
        logger.warning(f"Few morphology pairs found ({len(train_pairs)})")
        logger.info("Using general pairs with morphological patterns instead")
        # Fall back to all pairs if not enough morphology-specific
        train_pairs = all_pairs[:args.max_train]

    # Shuffle and limit
    random.seed(42)
    random.shuffle(train_pairs)
    train_pairs = train_pairs[:args.max_train]

    # Split for validation
    val_size = min(2000, len(train_pairs) // 10)
    val_pairs = train_pairs[:val_size]
    train_pairs = train_pairs[val_size:]

    logger.info(f"Train: {len(train_pairs)}, Val: {len(val_pairs)}")

    # Model selection
    # CAMeLBERT is encoder-only, so we use AraBART for seq2seq
    # In production, we'd create an EncoderDecoder from CAMeLBERT
    if args.use_arabart:
        model_name = 'moussaKam/AraBART'
        logger.info("Using AraBART (seq2seq)")
    else:
        # Try to use CAMeLBERT as encoder with decoder
        model_name = 'moussaKam/AraBART'  # Fallback to AraBART
        logger.info("Using AraBART as base (CAMeLBERT is encoder-only)")
        # Note: Full CAMeLBERT integration would require:
        # encoder = AutoModel.from_pretrained("CAMeL-Lab/bert-base-arabic-camelbert-mix")
        # decoder = create_decoder()
        # model = EncoderDecoderModel(encoder=encoder, decoder=decoder)

    logger.info(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    logger.info(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Datasets
    train_dataset = GECDataset(train_pairs, tokenizer)
    eval_dataset = GECDataset(val_pairs, tokenizer) if val_pairs else None

    # Output
    output_dir = OUTPUT_DIR / 'morphology_fixer'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training args
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=100,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_dataset else "no",
        predict_with_generate=True,
        fp16=False,
        report_to="none",
        save_total_limit=2,
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )

    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Train
    logger.info("Starting training...")
    start = datetime.now()
    trainer.train()
    duration = datetime.now() - start

    # Save
    trainer.save_model(str(output_dir / 'final'))
    tokenizer.save_pretrained(str(output_dir / 'final'))

    # Info
    info = {
        'model': 'morphology_fixer',
        'base': model_name,
        'epochs': args.epochs,
        'train_size': len(train_pairs),
        'val_size': len(val_pairs),
        'duration_minutes': duration.total_seconds() / 60,
        'timestamp': datetime.now().isoformat(),
    }
    with open(output_dir / 'training_info.json', 'w') as f:
        json.dump(info, f, indent=2)

    logger.info("=" * 60)
    logger.info("MORPHOLOGY FIXER TRAINING COMPLETE")
    logger.info(f"Duration: {duration}")
    logger.info(f"Saved to: {output_dir}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
