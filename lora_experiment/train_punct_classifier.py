#!/usr/bin/env python3
"""
Punct Classifier for Nahawi.
Binary token-level classification: "insert punct after this token?"

Uses frozen 124M encoder + linear classification head.
Trained only on QALB real data (no synthetic).
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import sentencepiece as spm
import json
import math
import os
from dataclasses import dataclass
from typing import List, Set


@dataclass
class PunctConfig:
    # Model
    base_model_path: str = "/home/ubuntu/nahawi/gec_clean/epoch_11.pt"
    tokenizer_path: str = "/home/ubuntu/nahawi/nahawi_spm.model"

    # Data (QALB only - real data)
    train_data: str = "/home/ubuntu/nahawi/data/qalb_real_train.json"
    dev_data: str = "/home/ubuntu/nahawi/data/qalb_real_dev.json"

    # Training
    output_dir: str = "/home/ubuntu/nahawi/punct_classifier"
    batch_size: int = 64
    learning_rate: float = 1e-4
    num_epochs: int = 5
    max_seq_len: int = 256

    # Punct tokens
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
    """
    Token-level punct insertion classifier.
    For each token, predicts which punct (if any) to insert after it.
    """

    def __init__(self, config: dict, num_punct_classes: int):
        super().__init__()

        # Embedding (frozen, loaded from base)
        self.embedding = nn.Embedding(config["vocab_size"], config["d_model"])
        self.pos_encoder = PositionalEncoding(config["d_model"], config["max_seq_len"])

        # Encoder (frozen, loaded from base)
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

        # Classification head (trainable)
        # num_punct_classes includes 0 = no punct
        self.classifier = nn.Sequential(
            nn.Linear(config["d_model"], config["d_model"] // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config["d_model"] // 2, num_punct_classes),
        )

    def freeze_encoder(self):
        """Freeze embedding and encoder, keep classifier trainable."""
        for param in self.embedding.parameters():
            param.requires_grad = False
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.pos_encoder.parameters():
            param.requires_grad = False

        # Count trainable
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total params: {total:,}")
        print(f"Trainable params: {trainable:,} ({100*trainable/total:.2f}%)")

    def forward(self, src_ids, src_mask=None):
        # Embed
        x = self.dropout(self.pos_encoder(self.embedding(src_ids) * math.sqrt(self.d_model)))

        # Encode
        encoded = self.encoder(x, src_key_padding_mask=src_mask)

        # Classify each token
        logits = self.classifier(encoded)

        return logits


class PunctDataset(Dataset):
    """
    Dataset for punct classification.
    Creates token-level labels from source->target pairs.
    """

    def __init__(self, data_path: str, tokenizer: spm.SentencePieceProcessor,
                 punct_chars: str, max_len: int = 256):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.punct_chars = set(punct_chars)

        # Build punct -> class mapping
        self.punct_to_class = {p: i + 1 for i, p in enumerate(punct_chars)}
        self.class_to_punct = {i + 1: p for i, p in enumerate(punct_chars)}
        self.num_classes = len(punct_chars) + 1  # +1 for no punct (class 0)

        self.data = []
        self._load_data(data_path)

    def _load_data(self, data_path):
        print(f"Loading punct data from {data_path}...")

        with open(data_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if content.startswith('['):
                items = json.loads(content)
            else:
                items = [json.loads(line) for line in content.split('\n') if line.strip()]

        for item in items:
            src = item.get('source', item.get('src', ''))
            tgt = item.get('target', item.get('tgt', ''))
            if not src or not tgt:
                continue

            # Create token-level labels from alignment
            labels = self._create_labels(src, tgt)
            if labels is not None:
                self.data.append((src, labels))

        print(f"  Loaded {len(self.data):,} examples")

    def _create_labels(self, source: str, target: str) -> List[int]:
        """
        Create punct insertion labels by aligning source and target.
        Returns list of punct class IDs (0 = no punct) for each source token.
        """
        # Tokenize source (without punct)
        src_tokens = self.tokenizer.encode(source)[:self.max_len - 2]

        # Tokenize target
        tgt_tokens = self.tokenizer.encode(target)[:self.max_len - 2]

        # Decode to get actual strings
        src_pieces = [self.tokenizer.id_to_piece(t) for t in src_tokens]
        tgt_pieces = [self.tokenizer.id_to_piece(t) for t in tgt_tokens]

        # Simple alignment: for each source token, check if target has punct after it
        labels = [0] * len(src_tokens)

        # Walk through target and find punct insertions
        src_idx = 0
        for tgt_idx, tgt_piece in enumerate(tgt_pieces):
            # Check if this is a punct token
            punct_found = None
            for p in self.punct_chars:
                if p in tgt_piece:
                    punct_found = p
                    break

            if punct_found:
                # Assign this punct to previous source token
                if src_idx > 0:
                    labels[src_idx - 1] = self.punct_to_class[punct_found]
            else:
                # Non-punct token, advance source pointer if matching
                if src_idx < len(src_pieces):
                    src_idx += 1

        return labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src, labels = self.data[idx]

        src_ids = self.tokenizer.encode(src)[:self.max_len - 2]
        src_ids = [2] + src_ids + [3]  # BOS, EOS

        # Adjust labels for BOS/EOS
        labels = [0] + labels[:len(src_ids) - 2] + [0]

        # Pad labels if needed
        while len(labels) < len(src_ids):
            labels.append(0)

        return {
            'src_ids': src_ids,
            'labels': labels,
        }


def collate_fn(batch, pad_id=0):
    src_ids = [torch.tensor(item['src_ids']) for item in batch]
    labels = [torch.tensor(item['labels']) for item in batch]

    src_ids = nn.utils.rnn.pad_sequence(src_ids, batch_first=True, padding_value=pad_id)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)  # -100 = ignore

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

        # Load tokenizer
        print("Loading tokenizer...")
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.load(config.tokenizer_path)

        # Model config
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

        # Number of punct classes
        num_punct_classes = len(config.punct_chars) + 1  # +1 for no punct

        # Create model
        print("Creating punct classifier...")
        self.model = PunctClassifier(model_config, num_punct_classes)

        # Load base encoder weights
        print(f"Loading base encoder from {config.base_model_path}...")
        checkpoint = torch.load(config.base_model_path, map_location='cpu', weights_only=False)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # Load only encoder weights
        encoder_state = {}
        for k, v in state_dict.items():
            if k.startswith('embedding.') or k.startswith('encoder.') or k.startswith('pos_encoder.'):
                encoder_state[k] = v

        self.model.load_state_dict(encoder_state, strict=False)

        # Freeze encoder
        self.model.freeze_encoder()
        self.model.to(self.device)

        # Dataset
        self.train_dataset = PunctDataset(
            config.train_data, self.tokenizer,
            config.punct_chars, config.max_seq_len
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=collate_fn,
        )

        # Optimizer
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(trainable_params, lr=config.learning_rate)

        # Scheduler
        total_steps = len(self.train_loader) * config.num_epochs
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps)

        # Loss (class imbalance: most tokens have no punct)
        # Weight punct classes higher
        class_weights = torch.ones(num_punct_classes)
        class_weights[0] = 0.1  # Reduce weight for "no punct"
        self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device), ignore_index=-100)

        # Output
        os.makedirs(config.output_dir, exist_ok=True)

    def train(self):
        print("\n" + "=" * 60)
        print("PUNCT CLASSIFIER TRAINING")
        print("=" * 60)
        print(f"Epochs: {self.config.num_epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"Punct classes: {self.config.punct_chars}")
        print()

        best_loss = float('inf')

        for epoch in range(self.config.num_epochs):
            self.model.train()
            epoch_loss = 0
            num_batches = 0

            for i, batch in enumerate(self.train_loader):
                src_ids = batch['src_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                src_mask = batch['src_mask'].to(self.device)

                # Forward
                logits = self.model(src_ids, src_mask)

                # Loss
                loss = self.criterion(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1)
                )

                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()

                epoch_loss += loss.item()
                num_batches += 1

                if (i + 1) % 100 == 0:
                    avg_loss = epoch_loss / num_batches
                    print(f"Epoch {epoch+1} | Batch {i+1}/{len(self.train_loader)} | Loss: {avg_loss:.4f}")

            # End of epoch
            avg_loss = epoch_loss / num_batches
            print(f"\nEpoch {epoch+1} complete | Avg Loss: {avg_loss:.4f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save_checkpoint("best_model.pt")
                print(f"  New best model saved!")

            self.save_checkpoint(f"epoch_{epoch+1}.pt")

        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)

    def save_checkpoint(self, filename):
        path = os.path.join(self.config.output_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'punct_to_class': self.train_dataset.punct_to_class,
            'class_to_punct': self.train_dataset.class_to_punct,
        }, path)
        print(f"  Saved {path}")


def main():
    config = PunctConfig()
    trainer = PunctTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
