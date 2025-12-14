#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train specialized AraBART models on filtered error types.
Creates focused models for specific error categories.
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
from typing import List, Tuple, Set
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
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)

# Paths
NAHAWI_DIR = Path('C:/nahawi')
QALB_DIR = NAHAWI_DIR / 'qalb_real_data'
SYNTHETIC_DIR = NAHAWI_DIR / 'synthetic_v4'
OUTPUT_DIR = NAHAWI_DIR / 'nahawi_ensemble' / 'checkpoints'

# Error type filters for specialized models
ERROR_FILTERS = {
    'hamza_fixer': {
        'patterns': [
            r'[أإآءؤئا]',  # Any hamza character
        ],
        'description': 'Hamza errors (أ/إ/ا/آ/ء/ؤ/ئ)',
        'min_diff_chars': {'أ', 'إ', 'آ', 'ء', 'ؤ', 'ئ', 'ا'},
    },
    'space_fixer': {
        'patterns': [
            r'\s',  # Space differences
        ],
        'description': 'Space/merge/split errors',
        'check_space_diff': True,
    },
    'spelling_fixer': {
        'patterns': [
            r'.',  # Any character (will filter by edit distance)
        ],
        'description': 'Spelling errors (char-level)',
        'max_edit_distance': 3,
        'min_edit_distance': 1,
    },
    'deleted_word_fixer': {
        'description': 'Missing/deleted word errors',
        'check_word_count': True,
    },
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


def fast_char_diff(s1: str, s2: str) -> int:
    """Fast character difference count (approximates edit distance)."""
    # Quick length check
    len_diff = abs(len(s1) - len(s2))
    if len_diff > 3:
        return len_diff

    # Count character mismatches
    diff = 0
    min_len = min(len(s1), len(s2))
    for i in range(min_len):
        if s1[i] != s2[i]:
            diff += 1
            if diff > 3:
                return diff
    return diff + len_diff


def has_hamza_diff(source: str, target: str) -> bool:
    """Check if difference involves hamza characters."""
    hamza_chars = set('أإآءؤئا')

    # Find character differences
    for i, (s, t) in enumerate(zip(source, target)):
        if s != t and (s in hamza_chars or t in hamza_chars):
            return True

    # Check length differences with hamza
    if len(source) != len(target):
        for c in hamza_chars:
            if source.count(c) != target.count(c):
                return True

    return False


def has_space_diff(source: str, target: str) -> bool:
    """Check if difference involves spaces (merge/split)."""
    source_words = source.split()
    target_words = target.split()
    return len(source_words) != len(target_words)


def has_deleted_word(source: str, target: str) -> bool:
    """Check if source is missing words that are in target (deletion error)."""
    source_words = source.split()
    target_words = target.split()
    # Source has fewer words than target = words were deleted from source
    return len(source_words) < len(target_words)


def filter_pairs_for_model(pairs: List[Tuple[str, str]], model_type: str) -> List[Tuple[str, str]]:
    """Filter pairs based on model specialization."""
    filtered = []

    for source, target in pairs:
        if source == target:
            continue

        if model_type == 'hamza_fixer':
            if has_hamza_diff(source, target):
                filtered.append((source, target))

        elif model_type == 'space_fixer':
            if has_space_diff(source, target):
                filtered.append((source, target))

        elif model_type == 'spelling_fixer':
            # Spelling: small edit distance, no space changes
            if not has_space_diff(source, target):
                dist = fast_char_diff(source, target)
                if 1 <= dist <= 3:
                    filtered.append((source, target))

        elif model_type == 'deleted_word_fixer':
            # Deleted words: source has fewer words than target
            if has_deleted_word(source, target):
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


def train_model(model_type: str, epochs: int, batch_size: int, max_train: int, lr: float):
    logger.info("=" * 60)
    logger.info(f"TRAINING: {model_type}")
    logger.info("=" * 60)

    # Load all data first
    all_pairs = []

    # QALB
    qalb_train = QALB_DIR / 'train.tsv'
    if qalb_train.exists():
        pairs = load_pairs(qalb_train)
        logger.info(f"QALB train: {len(pairs)} pairs")
        all_pairs.extend(pairs)

    # Synthetic
    synthetic_train = SYNTHETIC_DIR / 'train.tsv'
    if synthetic_train.exists():
        pairs = load_pairs(synthetic_train, max_train * 2)  # Load more to filter
        logger.info(f"Synthetic: {len(pairs)} pairs")
        all_pairs.extend(pairs)

    if not all_pairs:
        logger.error("No training data!")
        return

    # Filter for this model type
    logger.info(f"Filtering for {model_type}...")
    train_pairs = filter_pairs_for_model(all_pairs, model_type)
    logger.info(f"Filtered: {len(train_pairs)} relevant pairs")

    if len(train_pairs) < 100:
        logger.error(f"Not enough pairs for {model_type} (need >= 100)")
        return

    # Shuffle and limit
    random.seed(42)
    random.shuffle(train_pairs)
    train_pairs = train_pairs[:max_train]

    # Split for validation
    val_size = min(2000, len(train_pairs) // 10)
    val_pairs = train_pairs[:val_size]
    train_pairs = train_pairs[val_size:]

    logger.info(f"Train: {len(train_pairs)}, Val: {len(val_pairs)}")

    # Load model
    logger.info("Loading AraBART...")
    tokenizer = AutoTokenizer.from_pretrained('moussaKam/AraBART')
    model = AutoModelForSeq2SeqLM.from_pretrained('moussaKam/AraBART')

    logger.info(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Datasets
    train_dataset = GECDataset(train_pairs, tokenizer)
    eval_dataset = GECDataset(val_pairs, tokenizer) if val_pairs else None

    # Output
    output_dir = OUTPUT_DIR / model_type
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training args
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
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
        'model': model_type,
        'base': 'moussaKam/AraBART',
        'epochs': epochs,
        'train_size': len(train_pairs),
        'val_size': len(val_pairs),
        'duration_minutes': duration.total_seconds() / 60,
        'timestamp': datetime.now().isoformat(),
    }
    with open(output_dir / 'training_info.json', 'w') as f:
        json.dump(info, f, indent=2)

    logger.info("=" * 60)
    logger.info(f"{model_type} TRAINING COMPLETE")
    logger.info(f"Duration: {duration}")
    logger.info(f"Saved to: {output_dir}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        choices=['hamza_fixer', 'space_fixer', 'spelling_fixer', 'deleted_word_fixer', 'all'])
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--max-train', type=int, default=30000)
    parser.add_argument('--lr', type=float, default=2e-5)
    args = parser.parse_args()

    if args.model == 'all':
        for model_type in ['hamza_fixer', 'space_fixer', 'spelling_fixer', 'deleted_word_fixer']:
            train_model(model_type, args.epochs, args.batch_size, args.max_train, args.lr)
    else:
        train_model(args.model, args.epochs, args.batch_size, args.max_train, args.lr)


if __name__ == '__main__':
    main()
