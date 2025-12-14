#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple AraBART training script for GEC.
Trains on combined QALB + synthetic data.
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
from typing import List, Tuple
from datetime import datetime
import logging

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
    parser.add_argument('--model-name', type=str, default='general_gec')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--max-train', type=int, default=50000)
    parser.add_argument('--max-val', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=2e-5)
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info(f"TRAINING: {args.model_name}")
    logger.info("=" * 60)

    # Load data
    train_pairs = []
    val_pairs = []

    # QALB
    qalb_train = QALB_DIR / 'train.tsv'
    if qalb_train.exists():
        pairs = load_pairs(qalb_train)
        logger.info(f"QALB train: {len(pairs)} pairs")
        train_pairs.extend(pairs)

    qalb_dev = QALB_DIR / 'dev.tsv'
    if qalb_dev.exists():
        val_pairs = load_pairs(qalb_dev, args.max_val)
        logger.info(f"QALB dev: {len(val_pairs)} pairs")

    # Synthetic
    synthetic_train = SYNTHETIC_DIR / 'train.tsv'
    if synthetic_train.exists():
        pairs = load_pairs(synthetic_train, args.max_train)
        logger.info(f"Synthetic: {len(pairs)} pairs")
        train_pairs.extend(pairs)

    if not train_pairs:
        logger.error("No training data!")
        return

    # Shuffle and limit
    random.seed(42)
    random.shuffle(train_pairs)
    train_pairs = train_pairs[:args.max_train]

    logger.info(f"Total train: {len(train_pairs)}")
    logger.info(f"Total val: {len(val_pairs)}")

    # Load model
    logger.info("Loading AraBART...")
    tokenizer = AutoTokenizer.from_pretrained('moussaKam/AraBART')
    model = AutoModelForSeq2SeqLM.from_pretrained('moussaKam/AraBART')

    logger.info(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Datasets
    train_dataset = GECDataset(train_pairs, tokenizer)
    eval_dataset = GECDataset(val_pairs, tokenizer) if val_pairs else None

    # Output
    output_dir = OUTPUT_DIR / args.model_name
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
        fp16=False,  # CPU doesn't support fp16
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
        'model': args.model_name,
        'base': 'moussaKam/AraBART',
        'epochs': args.epochs,
        'train_size': len(train_pairs),
        'val_size': len(val_pairs),
        'duration_minutes': duration.total_seconds() / 60,
        'timestamp': datetime.now().isoformat(),
    }
    with open(output_dir / 'training_info.json', 'w') as f:
        json.dump(info, f, indent=2)

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info(f"Duration: {duration}")
    logger.info(f"Saved to: {output_dir}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
