#!/usr/bin/env python3
"""
LoRA Fine-tuning V2 for Nahawi 124M GEC Model.

Key changes from V1:
- Uses epoch_11 as base (best content model at 81% no-punct)
- 1:4 ratio (QALB 4x repeated + stratified synthetic)
- Rank 32, both attention and FFN layers
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


@dataclass
class LoRAConfig:
    # Model - using V15 (better on FASIH)
    base_model_path: str = "/home/ubuntu/nahawi/fasih_v15_model/best_model.pt"
    tokenizer_path: str = "/home/ubuntu/nahawi/nahawi_spm.model"

    # LoRA settings (per expert recommendation)
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05

    # Training data (1:4 ratio from sample_balanced_300k.py)
    train_data: str = "/home/ubuntu/nahawi/data/lora_train_300k.json"
    output_dir: str = "/home/ubuntu/nahawi/lora_model_v15"

    # Training params - MAX VRAM (96GB available, targeting 90%)
    batch_size: int = 192
    gradient_accumulation: int = 4  # Effective batch 768
    learning_rate: float = 2e-4
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    max_seq_len: int = 256

    # Evaluation
    eval_every: int = 1000
    save_every: int = 2000


class LoRALinear(nn.Module):
    """LoRA-augmented linear layer."""

    def __init__(self, original_layer: nn.Linear, rank: int, alpha: int, dropout: float = 0.0):
        super().__init__()
        self.original = original_layer
        self.rank = rank
        self.scaling = alpha / rank

        for param in self.original.parameters():
            param.requires_grad = False

        in_features = original_layer.in_features
        out_features = original_layer.out_features

        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    @property
    def weight(self):
        """Expose original weight for PyTorch internals."""
        return self.original.weight

    @property
    def bias(self):
        """Expose original bias for PyTorch internals."""
        return self.original.bias

    @property
    def in_features(self):
        return self.original.in_features

    @property
    def out_features(self):
        return self.original.out_features

    def forward(self, x):
        original_output = self.original(x)
        lora_output = (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return original_output + lora_output


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


class NahawiGECWithLoRA(nn.Module):
    """Nahawi GEC with LoRA adapters on attention + FFN."""

    def __init__(self, config: dict, lora_config: LoRAConfig):
        super().__init__()
        self.config = config
        self.lora_config = lora_config

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

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config["d_model"],
            nhead=config["nhead"],
            dim_feedforward=config["dim_feedforward"],
            dropout=config["dropout"],
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config["num_decoder_layers"])

        self.output_projection = nn.Linear(config["d_model"], config["vocab_size"])
        self.output_projection.weight = self.embedding.weight

        self.dropout = nn.Dropout(config["dropout"])
        self.d_model = config["d_model"]

    def add_lora_adapters(self):
        """Add LoRA adapters to attention and FFN layers."""
        cfg = self.lora_config

        # Freeze all base parameters
        for param in self.parameters():
            param.requires_grad = False

        # Add LoRA to encoder
        for layer in self.encoder.layers:
            layer.self_attn.out_proj = LoRALinear(
                layer.self_attn.out_proj, cfg.lora_rank, cfg.lora_alpha, cfg.lora_dropout
            )
            layer.linear1 = LoRALinear(
                layer.linear1, cfg.lora_rank, cfg.lora_alpha, cfg.lora_dropout
            )
            layer.linear2 = LoRALinear(
                layer.linear2, cfg.lora_rank, cfg.lora_alpha, cfg.lora_dropout
            )

        # Add LoRA to decoder
        for layer in self.decoder.layers:
            layer.self_attn.out_proj = LoRALinear(
                layer.self_attn.out_proj, cfg.lora_rank, cfg.lora_alpha, cfg.lora_dropout
            )
            layer.multihead_attn.out_proj = LoRALinear(
                layer.multihead_attn.out_proj, cfg.lora_rank, cfg.lora_alpha, cfg.lora_dropout
            )
            layer.linear1 = LoRALinear(
                layer.linear1, cfg.lora_rank, cfg.lora_alpha, cfg.lora_dropout
            )
            layer.linear2 = LoRALinear(
                layer.linear2, cfg.lora_rank, cfg.lora_alpha, cfg.lora_dropout
            )

        # Count params
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total params: {total:,}")
        print(f"Trainable params: {trainable:,} ({100*trainable/total:.2f}%)")

    def forward(self, src_ids, tgt_ids, src_mask=None, tgt_mask=None):
        src_emb = self.dropout(self.pos_encoder(self.embedding(src_ids) * math.sqrt(self.d_model)))
        tgt_emb = self.dropout(self.pos_encoder(self.embedding(tgt_ids) * math.sqrt(self.d_model)))

        tgt_len = tgt_ids.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len, device=tgt_ids.device)

        memory = self.encoder(src_emb, src_key_padding_mask=src_mask)
        output = self.decoder(
            tgt_emb, memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_mask,
            memory_key_padding_mask=src_mask,
        )

        return self.output_projection(output)

    @torch.no_grad()
    def generate(self, src_ids, max_len=256, eos_id=3):
        self.eval()
        device = src_ids.device
        batch_size = src_ids.size(0)

        src_emb = self.pos_encoder(self.embedding(src_ids) * math.sqrt(self.d_model))
        memory = self.encoder(src_emb)

        generated = torch.full((batch_size, 1), 2, dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            tgt_emb = self.pos_encoder(self.embedding(generated) * math.sqrt(self.d_model))
            causal_mask = nn.Transformer.generate_square_subsequent_mask(generated.size(1), device=device)
            output = self.decoder(tgt_emb, memory, tgt_mask=causal_mask)
            logits = self.output_projection(output[:, -1, :])
            next_token = logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if (next_token == eos_id).all():
                break

        return generated


class GECDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: spm.SentencePieceProcessor, max_len: int = 256):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = []

        print(f"Loading data from {data_path}...")
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    src = item.get('source', item.get('src', ''))
                    tgt = item.get('target', item.get('tgt', ''))
                    if src and tgt:
                        self.data.append((src, tgt))
                except:
                    pass

        print(f"  Loaded {len(self.data):,} pairs")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src, tgt = self.data[idx]
        src_ids = [2] + self.tokenizer.encode(src)[:self.max_len - 2] + [3]
        tgt_ids = [2] + self.tokenizer.encode(tgt)[:self.max_len - 2] + [3]
        return {'src_ids': src_ids, 'tgt_ids': tgt_ids}


def collate_fn(batch, pad_id=0):
    src_ids = [torch.tensor(item['src_ids']) for item in batch]
    tgt_ids = [torch.tensor(item['tgt_ids']) for item in batch]

    src_ids = nn.utils.rnn.pad_sequence(src_ids, batch_first=True, padding_value=pad_id)
    tgt_ids = nn.utils.rnn.pad_sequence(tgt_ids, batch_first=True, padding_value=pad_id)

    return {
        'src_ids': src_ids,
        'tgt_ids': tgt_ids,
        'src_mask': (src_ids == pad_id),
        'tgt_mask': (tgt_ids == pad_id),
    }


class LoRATrainer:
    def __init__(self, config: LoRAConfig):
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

        print("Creating model with LoRA...")
        self.model = NahawiGECWithLoRA(model_config, config)

        print(f"Loading base model from {config.base_model_path}...")
        checkpoint = torch.load(config.base_model_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        self.model.load_state_dict(state_dict, strict=False)

        self.model.add_lora_adapters()
        self.model.to(self.device)

        self.train_dataset = GECDataset(config.train_data, self.tokenizer, config.max_seq_len)
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=collate_fn,
        )

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(trainable_params, lr=config.learning_rate, weight_decay=0.01)

        total_steps = len(self.train_loader) * config.num_epochs // config.gradient_accumulation
        warmup_steps = int(total_steps * config.warmup_ratio)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps - warmup_steps)
        self.warmup_steps = warmup_steps

        self.criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
        os.makedirs(config.output_dir, exist_ok=True)

    def train(self):
        print("\n" + "=" * 60)
        print("LORA V2 TRAINING")
        print("=" * 60)
        print(f"Base model: epoch_11 (81% no-punct)")
        print(f"Data: 1:4 ratio (QALB 4x + synthetic)")
        print(f"LoRA rank: {self.config.lora_rank}")
        print(f"Epochs: {self.config.num_epochs}")
        print(f"Effective batch: {self.config.batch_size * self.config.gradient_accumulation}")
        print()

        global_step = 0
        best_loss = float('inf')

        for epoch in range(self.config.num_epochs):
            self.model.train()
            epoch_loss = 0
            num_batches = 0

            for i, batch in enumerate(self.train_loader):
                src_ids = batch['src_ids'].to(self.device)
                tgt_ids = batch['tgt_ids'].to(self.device)
                src_mask = batch['src_mask'].to(self.device)
                tgt_mask = batch['tgt_mask'].to(self.device)

                logits = self.model(src_ids, tgt_ids[:, :-1], src_mask, tgt_mask[:, :-1])
                loss = self.criterion(
                    logits.reshape(-1, logits.size(-1)),
                    tgt_ids[:, 1:].reshape(-1)
                )

                loss = loss / self.config.gradient_accumulation
                loss.backward()

                epoch_loss += loss.item() * self.config.gradient_accumulation
                num_batches += 1

                if (i + 1) % self.config.gradient_accumulation == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                    if global_step >= self.warmup_steps:
                        self.scheduler.step()

                    self.optimizer.zero_grad()
                    global_step += 1

                    if global_step % 100 == 0:
                        avg_loss = epoch_loss / num_batches
                        lr = self.optimizer.param_groups[0]['lr']
                        print(f"Epoch {epoch+1} | Step {global_step} | Loss: {avg_loss:.4f} | LR: {lr:.2e}")

                    if global_step % self.config.save_every == 0:
                        self.save_checkpoint(f"checkpoint_{global_step}.pt")

            avg_loss = epoch_loss / num_batches
            print(f"\nEpoch {epoch+1} complete | Avg Loss: {avg_loss:.4f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save_checkpoint("best_model.pt")
                print(f"  New best model saved!")

            self.save_checkpoint(f"epoch_{epoch+1}.pt")

        print("\n" + "=" * 60)
        print("LORA TRAINING COMPLETE")
        print("=" * 60)

    def save_checkpoint(self, filename):
        path = os.path.join(self.config.output_dir, filename)

        lora_state = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                lora_state[name] = param.data.cpu()

        torch.save({
            'lora_state_dict': lora_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
        }, path)
        print(f"  Saved {path}")


def main():
    config = LoRAConfig()
    trainer = LoRATrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
