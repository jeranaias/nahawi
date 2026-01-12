#!/usr/bin/env python3
"""
LoRA Round 3: Fresh adapter on V15+Hamza epoch 1, trained on:
- 60K targeted pairs (verb conjugation, hamza swap, case endings)
- 20K QALB anchor (prevent forgetting)

Rank 64, Alpha 128, 1 epoch first.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

# For unpickling the hamza checkpoint
@dataclass
class LoRAConfig:
    base_model_path: str = ""
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    train_data: str = ""
    dev_data: str = ""
    output_dir: str = ""
    batch_size: int = 16
    learning_rate: float = 2e-4
    num_epochs: int = 3
    warmup_steps: int = 500
    max_seq_len: int = 256
    gradient_accumulation_steps: int = 1
    save_steps: int = 1000
    eval_steps: int = 500
    logging_steps: int = 100

@dataclass
class Config:
    # Model
    base_model_path: str = "/home/ubuntu/nahawi/fasih_v15_model/best_model.pt"
    hamza_lora_path: str = "/home/ubuntu/nahawi/lora_model_hamza/epoch_1.pt"
    tokenizer_path: str = "/home/ubuntu/nahawi/nahawi_spm.model"

    # Data
    targeted_data: str = "/home/ubuntu/nahawi/data/targeted_fixes_train.json"
    qalb_data: str = "/home/ubuntu/nahawi/data/qalb_real_train.json"
    qalb_sample_size: int = 20000

    # LoRA
    lora_rank: int = 64
    lora_alpha: int = 128

    # Training
    batch_size: int = 16
    learning_rate: float = 2e-4
    num_epochs: int = 1
    max_seq_len: int = 256

    # Output
    output_dir: str = "/home/ubuntu/nahawi/lora_model_round3"


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class LoRALinear(nn.Module):
    def __init__(self, original, rank, alpha):
        super().__init__()
        self.original = original
        self.rank = rank
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.zeros(rank, original.in_features))
        self.lora_B = nn.Parameter(torch.zeros(original.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # lora_B stays zero-initialized

    @property
    def weight(self):
        return self.original.weight
    @property
    def bias(self):
        return self.original.bias

    def forward(self, x):
        return self.original(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scaling


class NahawiGEC(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config["vocab_size"], config["d_model"])
        self.pos_encoder = PositionalEncoding(config["d_model"], config["max_seq_len"])
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config["d_model"], nhead=config["nhead"],
            dim_feedforward=config["dim_feedforward"], dropout=config["dropout"], batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config["num_encoder_layers"])
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config["d_model"], nhead=config["nhead"],
            dim_feedforward=config["dim_feedforward"], dropout=config["dropout"], batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config["num_decoder_layers"])
        self.output_projection = nn.Linear(config["d_model"], config["vocab_size"])
        self.output_projection.weight = self.embedding.weight
        self.d_model = config["d_model"]

    def add_lora(self, rank=64, alpha=128):
        for layer in self.encoder.layers:
            layer.self_attn.out_proj = LoRALinear(layer.self_attn.out_proj, rank, alpha)
            layer.linear1 = LoRALinear(layer.linear1, rank, alpha)
            layer.linear2 = LoRALinear(layer.linear2, rank, alpha)
        for layer in self.decoder.layers:
            layer.self_attn.out_proj = LoRALinear(layer.self_attn.out_proj, rank, alpha)
            layer.multihead_attn.out_proj = LoRALinear(layer.multihead_attn.out_proj, rank, alpha)
            layer.linear1 = LoRALinear(layer.linear1, rank, alpha)
            layer.linear2 = LoRALinear(layer.linear2, rank, alpha)

    def forward(self, src_ids, tgt_ids):
        # Embed
        src_emb = self.pos_encoder(self.embedding(src_ids) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.embedding(tgt_ids) * math.sqrt(self.d_model))

        # Encode
        memory = self.encoder(src_emb)

        # Decode with causal mask
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_ids.size(1), device=tgt_ids.device)
        output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)

        # Project to vocab
        logits = self.output_projection(output)
        return logits


class GECDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        src = item.get('source', item.get('src', ''))
        tgt = item.get('target', item.get('tgt', ''))

        src_ids = [2] + self.tokenizer.encode(src)[:self.max_len-2] + [3]
        tgt_ids = [2] + self.tokenizer.encode(tgt)[:self.max_len-2] + [3]

        return {
            'src_ids': torch.tensor(src_ids),
            'tgt_ids': torch.tensor(tgt_ids)
        }


def collate_fn(batch):
    src_ids = [item['src_ids'] for item in batch]
    tgt_ids = [item['tgt_ids'] for item in batch]

    max_src = max(len(s) for s in src_ids)
    max_tgt = max(len(t) for t in tgt_ids)

    src_padded = torch.zeros(len(batch), max_src, dtype=torch.long)
    tgt_padded = torch.zeros(len(batch), max_tgt, dtype=torch.long)

    for i, (s, t) in enumerate(zip(src_ids, tgt_ids)):
        src_padded[i, :len(s)] = s
        tgt_padded[i, :len(t)] = t

    return {'src_ids': src_padded, 'tgt_ids': tgt_padded}


def get_lora_params(model):
    """Get only LoRA parameters for optimization."""
    lora_params = []
    for name, param in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            lora_params.append(param)
    return lora_params


def save_lora_state(model, path):
    """Save only LoRA weights."""
    lora_state = {}
    for name, param in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            lora_state[name] = param.data.clone()
    torch.save({'lora_state_dict': lora_state}, path)


def main():
    cfg = Config()

    print("=" * 70)
    print("LORA ROUND 3: TARGETED FIXES")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create output dir
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(cfg.tokenizer_path)

    # Load model config
    model_config = {
        "vocab_size": 32000, "d_model": 768, "nhead": 12,
        "num_encoder_layers": 6, "num_decoder_layers": 6,
        "dim_feedforward": 3072, "dropout": 0.1, "max_seq_len": 256
    }

    # Build model
    print("\nBuilding model...")
    model = NahawiGEC(model_config)

    # Load V15 base
    print("  Loading V15 base weights...")
    base_ckpt = torch.load(cfg.base_model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(base_ckpt.get('model_state_dict', base_ckpt), strict=False)

    # Add LoRA (fresh, not loading hamza weights - starting from base + new LoRA)
    print(f"  Adding fresh LoRA adapters (rank={cfg.lora_rank}, alpha={cfg.lora_alpha})...")
    model.add_lora(rank=cfg.lora_rank, alpha=cfg.lora_alpha)

    # Load hamza LoRA weights into the model first, then we'll train on top
    print("  Loading Hamza LoRA epoch 1 weights...")
    hamza_ckpt = torch.load(cfg.hamza_lora_path, map_location='cpu', weights_only=False)
    hamza_state = hamza_ckpt.get('lora_state_dict', {})
    model.load_state_dict(hamza_state, strict=False)

    model.to(device)

    # Freeze base, only train LoRA
    for name, param in model.named_parameters():
        if 'lora_A' not in name and 'lora_B' not in name:
            param.requires_grad = False

    lora_params = get_lora_params(model)
    print(f"  LoRA parameters: {sum(p.numel() for p in lora_params):,}")

    # Load training data
    print("\nLoading training data...")

    # Targeted fixes (60K)
    with open(cfg.targeted_data, 'r', encoding='utf-8') as f:
        targeted_data = json.load(f)
    print(f"  Targeted fixes: {len(targeted_data):,}")

    # QALB anchor (sample 20K)
    with open(cfg.qalb_data, 'r', encoding='utf-8') as f:
        qalb_data = json.load(f)
    random.shuffle(qalb_data)
    qalb_sample = qalb_data[:cfg.qalb_sample_size]
    print(f"  QALB anchor: {len(qalb_sample):,}")

    # Combine
    all_data = targeted_data + qalb_sample
    random.shuffle(all_data)
    print(f"  Total: {len(all_data):,}")

    # Create dataloader
    dataset = GECDataset(all_data, tokenizer, cfg.max_seq_len)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True,
                           collate_fn=collate_fn, num_workers=2)

    # Optimizer
    optimizer = optim.AdamW(lora_params, lr=cfg.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # Training
    print(f"\nTraining for {cfg.num_epochs} epoch(s)...")
    print(f"  Batches per epoch: {len(dataloader)}")

    model.train()

    for epoch in range(cfg.num_epochs):
        total_loss = 0

        for i, batch in enumerate(dataloader):
            src_ids = batch['src_ids'].to(device)
            tgt_ids = batch['tgt_ids'].to(device)

            # Forward
            logits = model(src_ids, tgt_ids[:, :-1])

            # Loss
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_ids[:, 1:].reshape(-1))

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
            optimizer.step()

            total_loss += loss.item()

            if (i + 1) % 500 == 0:
                avg_loss = total_loss / (i + 1)
                print(f"  [{i+1}/{len(dataloader)}] loss: {avg_loss:.4f}", flush=True)

        avg_loss = total_loss / len(dataloader)
        print(f"\nEpoch {epoch + 1} complete. Avg loss: {avg_loss:.4f}")

        # Save checkpoint
        save_path = Path(cfg.output_dir) / f"epoch_{epoch + 1}.pt"
        save_lora_state(model, save_path)
        print(f"Saved: {save_path}")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Checkpoints saved to: {cfg.output_dir}")
    print("Run eval_lora_round3.py to evaluate")


if __name__ == "__main__":
    main()
