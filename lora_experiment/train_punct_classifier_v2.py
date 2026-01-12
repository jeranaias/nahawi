#!/usr/bin/env python3
"""
Punct Classifier V2 - Trained on QALB + 5.9GB MSA Corpus.

Uses frozen epoch_11 encoder + classification head.
Learns "where does punct naturally occur in well-written Arabic."
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import sentencepiece as spm
import json
import math
import os
import random
from dataclasses import dataclass
from typing import List, Set, Iterator


@dataclass
class PunctConfig:
    # Model - V15 encoder (frozen for punct classification)
    base_model_path: str = "/home/ubuntu/nahawi/fasih_v15_model/best_model.pt"
    tokenizer_path: str = "/home/ubuntu/nahawi/nahawi_spm.model"

    # Data
    qalb_train: str = "/home/ubuntu/nahawi/data/qalb_real_train.json"
    qalb_dev: str = "/home/ubuntu/nahawi/data/qalb_real_dev.json"
    msa_corpus: str = "/home/ubuntu/nahawi/corpus/combined/msa_corpus_full.txt"

    # Training - MAX GPU (96GB VRAM)
    output_dir: str = "/home/ubuntu/nahawi/punct_classifier_v15"
    batch_size: int = 5000  # ~70% VRAM (safe)
    learning_rate: float = 1e-4
    num_epochs: int = 3
    max_seq_len: int = 256
    msa_samples_per_epoch: int = 500000  # Sample from 5.9GB corpus

    # Punct tokens (Arabic + standard)
    punct_chars: str = "،؛؟!.,:;?"


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class PunctClassifier(nn.Module):
    """Token-level punct classifier with frozen encoder."""

    def __init__(self, config: dict, num_punct_classes: int):
        super().__init__()

        self.embedding = nn.Embedding(config["vocab_size"], config["d_model"])
        self.pos_encoder = PositionalEncoding(config["d_model"], config["max_seq_len"])

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config["d_model"],
            nhead=config["nhead"],
            dim_feedforward=config["dim_feedforward"],
            dropout=config["dropout"],
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config["num_encoder_layers"])

        self.dropout = nn.Dropout(config["dropout"])
        self.d_model = config["d_model"]

        # Classification head with more capacity
        self.classifier = nn.Sequential(
            nn.Linear(config["d_model"], config["d_model"]),
            nn.LayerNorm(config["d_model"]),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config["d_model"], config["d_model"] // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config["d_model"] // 2, num_punct_classes),
        )

    def freeze_encoder(self):
        """Freeze embedding and encoder."""
        for param in self.embedding.parameters():
            param.requires_grad = False
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.pos_encoder.parameters():
            param.requires_grad = False

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total params: {total:,}")
        print(f"Trainable params: {trainable:,} ({100*trainable/total:.2f}%)")

    def forward(self, src_ids, src_mask=None):
        x = self.dropout(self.pos_encoder(self.embedding(src_ids) * math.sqrt(self.d_model)))
        encoded = self.encoder(x, src_key_padding_mask=src_mask)
        return self.classifier(encoded)


class MSACorpusDataset(IterableDataset):
    """
    Streaming dataset from large MSA corpus.
    Creates punct labels from existing punctuation in clean text.
    """

    def __init__(self, corpus_path: str, tokenizer: spm.SentencePieceProcessor,
                 punct_chars: str, max_len: int, samples_per_epoch: int):
        self.corpus_path = corpus_path
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples_per_epoch = samples_per_epoch

        self.punct_chars = set(punct_chars)
        self.punct_to_class = {p: i + 1 for i, p in enumerate(punct_chars)}
        self.num_classes = len(punct_chars) + 1

    def __iter__(self) -> Iterator[dict]:
        """Stream samples from corpus."""
        samples_yielded = 0

        while samples_yielded < self.samples_per_epoch:
            # Random seek into file
            with open(self.corpus_path, 'r', encoding='utf-8', errors='ignore') as f:
                # Get file size
                f.seek(0, 2)
                file_size = f.tell()

                # Random position
                pos = random.randint(0, max(0, file_size - 10000))
                f.seek(pos)

                # Skip partial line
                f.readline()

                # Read lines
                for line in f:
                    if samples_yielded >= self.samples_per_epoch:
                        break

                    line = line.strip()
                    if len(line) < 20:
                        continue

                    # Check if has punct (we only want punctuated sentences)
                    if not any(c in line for c in self.punct_chars):
                        continue

                    # Create sample
                    sample = self._create_sample(line)
                    if sample is not None:
                        yield sample
                        samples_yielded += 1

    def _create_sample(self, text: str) -> dict:
        """Create punct classification sample from text with existing punct."""
        # Tokenize
        tokens = self.tokenizer.encode(text)[:self.max_len - 2]
        if len(tokens) < 5:
            return None

        # Decode to pieces
        pieces = [self.tokenizer.id_to_piece(t) for t in tokens]

        # Create labels: for each token, what punct follows it?
        labels = []
        for i, piece in enumerate(pieces):
            # Check if next piece starts with punct or this piece ends with punct
            label = 0

            # Check if this piece ends with punct
            for p in self.punct_chars:
                if piece.endswith(p):
                    label = self.punct_to_class[p]
                    break

            # Check if next piece is punct-only
            if label == 0 and i + 1 < len(pieces):
                next_piece = pieces[i + 1]
                if len(next_piece) == 1 and next_piece in self.punct_chars:
                    label = self.punct_to_class[next_piece]

            labels.append(label)

        # Add BOS/EOS
        src_ids = [2] + tokens + [3]
        labels = [0] + labels + [0]

        return {
            'src_ids': src_ids,
            'labels': labels,
        }


class QALBPunctDataset(Dataset):
    """QALB dataset for punct learning (from target corrections)."""

    def __init__(self, data_path: str, tokenizer: spm.SentencePieceProcessor,
                 punct_chars: str, max_len: int):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.punct_chars = set(punct_chars)
        self.punct_to_class = {p: i + 1 for i, p in enumerate(punct_chars)}
        self.num_classes = len(punct_chars) + 1

        self.data = []
        self._load_data(data_path)

    def _load_data(self, data_path):
        print(f"Loading QALB punct data from {data_path}...")

        with open(data_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if content.startswith('['):
                items = json.loads(content)
            else:
                items = [json.loads(line) for line in content.split('\n') if line.strip()]

        for item in items:
            # Use TARGET (corrected) text - this has proper punct
            tgt = item.get('target', item.get('tgt', ''))
            if not tgt:
                continue

            # Only use sentences with punct
            if not any(c in tgt for c in self.punct_chars):
                continue

            sample = self._create_sample(tgt)
            if sample is not None:
                self.data.append(sample)

        print(f"  Loaded {len(self.data):,} punct examples")

    def _create_sample(self, text: str) -> dict:
        """Create punct sample from corrected text."""
        tokens = self.tokenizer.encode(text)[:self.max_len - 2]
        if len(tokens) < 5:
            return None

        pieces = [self.tokenizer.id_to_piece(t) for t in tokens]

        labels = []
        for i, piece in enumerate(pieces):
            label = 0

            for p in self.punct_chars:
                if piece.endswith(p):
                    label = self.punct_to_class[p]
                    break

            if label == 0 and i + 1 < len(pieces):
                next_piece = pieces[i + 1]
                if len(next_piece) == 1 and next_piece in self.punct_chars:
                    label = self.punct_to_class[next_piece]

            labels.append(label)

        src_ids = [2] + tokens + [3]
        labels = [0] + labels + [0]

        return {
            'src_ids': src_ids,
            'labels': labels,
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch, pad_id=0):
    src_ids = [torch.tensor(item['src_ids']) for item in batch]
    labels = [torch.tensor(item['labels']) for item in batch]

    src_ids = nn.utils.rnn.pad_sequence(src_ids, batch_first=True, padding_value=pad_id)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

    src_mask = (src_ids == pad_id)

    return {
        'src_ids': src_ids,
        'labels': labels,
        'src_mask': src_mask,
    }


class PunctTrainer:
    def __init__(self, config: PunctConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print("Loading tokenizer...")
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.load(config.tokenizer_path)

        model_config = {
            "vocab_size": 32000,
            "d_model": 768,
            "nhead": 12,
            "num_encoder_layers": 6,
            "num_decoder_layers": 6,
            "dim_feedforward": 3072,
            "dropout": 0.1,
            "max_seq_len": 256,
        }

        num_punct_classes = len(config.punct_chars) + 1
        print(f"Punct classes: {num_punct_classes} (0=none, 1-{num_punct_classes-1}=punct)")

        print("Creating punct classifier...")
        self.model = PunctClassifier(model_config, num_punct_classes)

        print(f"Loading base encoder from {config.base_model_path}...")
        checkpoint = torch.load(config.base_model_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint)

        encoder_state = {}
        for k, v in state_dict.items():
            if k.startswith('embedding.') or k.startswith('encoder.') or k.startswith('pos_encoder.'):
                encoder_state[k] = v

        self.model.load_state_dict(encoder_state, strict=False)
        print("  V15 encoder loaded (frozen for punct classification)")
        self.model.freeze_encoder()
        self.model.to(self.device)

        # Create datasets
        print("\nPreparing datasets...")

        # QALB dataset
        self.qalb_dataset = QALBPunctDataset(
            config.qalb_train, self.tokenizer,
            config.punct_chars, config.max_seq_len
        )

        # MSA corpus dataset (streaming)
        self.msa_dataset = MSACorpusDataset(
            config.msa_corpus, self.tokenizer,
            config.punct_chars, config.max_seq_len,
            config.msa_samples_per_epoch
        )

        # Combined loader (QALB repeated + MSA streamed)
        print(f"QALB examples: {len(self.qalb_dataset):,}")
        print(f"MSA samples per epoch: {config.msa_samples_per_epoch:,}")

        # Optimizer
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(trainable_params, lr=config.learning_rate, weight_decay=0.01)

        # Loss with class weighting
        class_weights = torch.ones(num_punct_classes)
        class_weights[0] = 0.1  # Much lower weight for "no punct"
        self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device), ignore_index=-100)

        os.makedirs(config.output_dir, exist_ok=True)

    def train(self):
        print("\n" + "=" * 60)
        print("PUNCT CLASSIFIER V2 TRAINING")
        print("=" * 60)
        print(f"Data: QALB ({len(self.qalb_dataset):,}) + MSA corpus ({self.config.msa_samples_per_epoch:,}/epoch)")
        print(f"Epochs: {self.config.num_epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print()

        best_loss = float('inf')

        for epoch in range(self.config.num_epochs):
            self.model.train()
            epoch_loss = 0
            num_batches = 0

            # Train on QALB (repeated throughout epoch)
            qalb_loader = DataLoader(
                self.qalb_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=2,
                collate_fn=collate_fn,
            )

            print(f"\nEpoch {epoch+1} - Training on QALB...")
            for i, batch in enumerate(qalb_loader):
                loss = self._train_step(batch)
                epoch_loss += loss
                num_batches += 1

                if (i + 1) % 5 == 0 or (i + 1) == len(qalb_loader):
                    print(f"  QALB [{i+1}/{len(qalb_loader)}] Loss: {epoch_loss/num_batches:.4f}")

            # Train on MSA corpus
            print(f"\nEpoch {epoch+1} - Training on MSA corpus...")
            msa_loader = DataLoader(
                self.msa_dataset,
                batch_size=self.config.batch_size,
                num_workers=4,
                collate_fn=collate_fn,
            )

            msa_batches = 0
            for i, batch in enumerate(msa_loader):
                loss = self._train_step(batch)
                epoch_loss += loss
                num_batches += 1
                msa_batches += 1

                if (i + 1) % 10 == 0:
                    print(f"  MSA [{i+1}] Loss: {epoch_loss/num_batches:.4f}")

                if msa_batches >= self.config.msa_samples_per_epoch // self.config.batch_size:
                    break

            # End of epoch
            avg_loss = epoch_loss / num_batches
            print(f"\nEpoch {epoch+1} complete | Avg Loss: {avg_loss:.4f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save_checkpoint("best_model.pt")
                print(f"  New best model saved!")

            self.save_checkpoint(f"epoch_{epoch+1}.pt")

        print("\n" + "=" * 60)
        print("PUNCT CLASSIFIER TRAINING COMPLETE")
        print("=" * 60)

    def _train_step(self, batch):
        src_ids = batch['src_ids'].to(self.device)
        labels = batch['labels'].to(self.device)
        src_mask = batch['src_mask'].to(self.device)

        self.optimizer.zero_grad()

        logits = self.model(src_ids, src_mask)
        loss = self.criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def save_checkpoint(self, filename):
        path = os.path.join(self.config.output_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'punct_to_class': self.qalb_dataset.punct_to_class,
            'class_to_punct': {v: k for k, v in self.qalb_dataset.punct_to_class.items()},
        }, path)
        print(f"  Saved {path}")


def main():
    config = PunctConfig()
    trainer = PunctTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
