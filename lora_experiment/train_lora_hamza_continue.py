#!/usr/bin/env python3
"""
Continue Hamza LoRA training from epoch 1 for epochs 2-3.
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
    base_model_path: str = "/home/ubuntu/nahawi/fasih_v15_model/best_model.pt"
    epoch1_lora_path: str = "/home/ubuntu/nahawi/lora_model_hamza/epoch_1.pt"
    tokenizer_path: str = "/home/ubuntu/nahawi/nahawi_spm.model"
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    train_data: str = "/home/ubuntu/nahawi/data/lora_train_hamza_aug.json"
    output_dir: str = "/home/ubuntu/nahawi/lora_model_hamza"
    batch_size: int = 192
    gradient_accumulation: int = 4
    learning_rate: float = 1e-4  # Lower LR for continued training
    num_epochs: int = 2  # Just epochs 2 and 3
    start_epoch: int = 2
    max_seq_len: int = 256


class LoRALinear(nn.Module):
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
        return self.original.weight

    @property
    def bias(self):
        return self.original.bias

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
    def __init__(self, config: dict, lora_config: LoRAConfig):
        super().__init__()
        self.config = config
        self.lora_config = lora_config
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
        self.dropout = nn.Dropout(config["dropout"])
        self.d_model = config["d_model"]

    def add_lora_adapters(self):
        lora_cfg = self.lora_config
        for param in self.parameters():
            param.requires_grad = False
        for layer in self.encoder.layers:
            layer.self_attn.out_proj = LoRALinear(layer.self_attn.out_proj, lora_cfg.lora_rank, lora_cfg.lora_alpha, lora_cfg.lora_dropout)
            layer.linear1 = LoRALinear(layer.linear1, lora_cfg.lora_rank, lora_cfg.lora_alpha, lora_cfg.lora_dropout)
            layer.linear2 = LoRALinear(layer.linear2, lora_cfg.lora_rank, lora_cfg.lora_alpha, lora_cfg.lora_dropout)
        for layer in self.decoder.layers:
            layer.self_attn.out_proj = LoRALinear(layer.self_attn.out_proj, lora_cfg.lora_rank, lora_cfg.lora_alpha, lora_cfg.lora_dropout)
            layer.multihead_attn.out_proj = LoRALinear(layer.multihead_attn.out_proj, lora_cfg.lora_rank, lora_cfg.lora_alpha, lora_cfg.lora_dropout)
            layer.linear1 = LoRALinear(layer.linear1, lora_cfg.lora_rank, lora_cfg.lora_alpha, lora_cfg.lora_dropout)
            layer.linear2 = LoRALinear(layer.linear2, lora_cfg.lora_rank, lora_cfg.lora_alpha, lora_cfg.lora_dropout)
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total params: {total:,}", flush=True)
        print(f"Trainable params: {trainable:,} ({100*trainable/total:.2f}%)", flush=True)

    def forward(self, src_ids, tgt_ids, src_mask=None, tgt_mask=None):
        src_emb = self.dropout(self.pos_encoder(self.embedding(src_ids) * math.sqrt(self.d_model)))
        tgt_emb = self.dropout(self.pos_encoder(self.embedding(tgt_ids) * math.sqrt(self.d_model)))
        tgt_len = tgt_ids.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len, device=tgt_ids.device)
        memory = self.encoder(src_emb, src_key_padding_mask=src_mask)
        output = self.decoder(tgt_emb, memory, tgt_mask=causal_mask, tgt_key_padding_mask=tgt_mask, memory_key_padding_mask=src_mask)
        return self.output_projection(output)


class GECDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_len: int = 256):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = []
        print(f"Loading data from {data_path}...", flush=True)
        with open(data_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if content.startswith('['):
                items = json.loads(content)
                for item in items:
                    src = item.get('source', item.get('src', ''))
                    tgt = item.get('target', item.get('tgt', ''))
                    if src and tgt:
                        self.data.append((src, tgt))
            else:
                for line in content.split('\n'):
                    try:
                        item = json.loads(line.strip())
                        src = item.get('source', item.get('src', ''))
                        tgt = item.get('target', item.get('tgt', ''))
                        if src and tgt:
                            self.data.append((src, tgt))
                    except:
                        pass
        print(f"  Loaded {len(self.data):,} pairs", flush=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src, tgt = self.data[idx]
        src_ids = self.tokenizer.encode(src)[:self.max_len - 2]
        tgt_ids = self.tokenizer.encode(tgt)[:self.max_len - 2]
        src_ids = [2] + src_ids + [3]
        tgt_ids = [2] + tgt_ids + [3]
        return {'src_ids': src_ids, 'tgt_ids': tgt_ids}


def collate_fn(batch, pad_id=0):
    src_ids = [torch.tensor(item['src_ids']) for item in batch]
    tgt_ids = [torch.tensor(item['tgt_ids']) for item in batch]
    src_ids = nn.utils.rnn.pad_sequence(src_ids, batch_first=True, padding_value=pad_id)
    tgt_ids = nn.utils.rnn.pad_sequence(tgt_ids, batch_first=True, padding_value=pad_id)
    src_mask = (src_ids == pad_id)
    tgt_mask = (tgt_ids == pad_id)
    return {'src_ids': src_ids, 'tgt_ids': tgt_ids, 'src_mask': src_mask, 'tgt_mask': tgt_mask}


def main():
    config = LoRAConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)

    print("\nLoading tokenizer...", flush=True)
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(config.tokenizer_path)

    model_config = {
        "vocab_size": 32000, "d_model": 768, "nhead": 12,
        "num_encoder_layers": 6, "num_decoder_layers": 6,
        "dim_feedforward": 3072, "dropout": 0.1, "max_seq_len": 256,
    }

    print("\nCreating model...", flush=True)
    model = NahawiGECWithLoRA(model_config, config)

    # Load base weights
    print(f"Loading base model from {config.base_model_path}...", flush=True)
    base_ckpt = torch.load(config.base_model_path, map_location='cpu', weights_only=False)
    state_dict = base_ckpt.get('model_state_dict', base_ckpt)
    model.load_state_dict(state_dict, strict=False)

    # Add LoRA adapters
    model.add_lora_adapters()

    # Load epoch 1 LoRA weights
    print(f"Loading epoch 1 LoRA weights from {config.epoch1_lora_path}...", flush=True)
    lora_ckpt = torch.load(config.epoch1_lora_path, map_location='cpu', weights_only=False)
    model.load_state_dict(lora_ckpt.get('lora_state_dict', {}), strict=False)
    print("  Epoch 1 weights loaded", flush=True)

    model.to(device)

    # Dataset
    train_dataset = GECDataset(config.train_data, tokenizer, config.max_seq_len)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn, pin_memory=True)

    # Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=config.learning_rate, weight_decay=0.01)

    # Scheduler
    total_steps = len(train_loader) * config.num_epochs // config.gradient_accumulation
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    criterion = nn.CrossEntropyLoss(ignore_index=0)

    print("\n" + "=" * 70, flush=True)
    print("CONTINUING HAMZA LORA TRAINING - EPOCHS 2-3", flush=True)
    print("=" * 70, flush=True)
    print(f"Starting from: Epoch 1 checkpoint", flush=True)
    print(f"Epochs to train: {config.num_epochs}", flush=True)
    print(f"Learning rate: {config.learning_rate}", flush=True)
    print(f"Training samples: {len(train_dataset):,}", flush=True)
    print(flush=True)

    global_step = 0
    best_loss = 0.4096  # Epoch 1 loss

    for epoch_offset in range(config.num_epochs):
        epoch = config.start_epoch + epoch_offset
        model.train()
        epoch_loss = 0
        num_batches = 0

        for i, batch in enumerate(train_loader):
            src_ids = batch['src_ids'].to(device)
            tgt_ids = batch['tgt_ids'].to(device)
            src_mask = batch['src_mask'].to(device)
            tgt_mask = batch['tgt_mask'].to(device)

            logits = model(src_ids, tgt_ids[:, :-1], src_mask, tgt_mask[:, :-1])
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_ids[:, 1:].reshape(-1))

            loss = loss / config.gradient_accumulation
            loss.backward()

            epoch_loss += loss.item() * config.gradient_accumulation
            num_batches += 1

            if (i + 1) % config.gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 100 == 0:
                    avg_loss = epoch_loss / num_batches
                    lr = scheduler.get_last_lr()[0]
                    print(f"Epoch {epoch} | Step {global_step} | Loss: {avg_loss:.4f} | LR: {lr:.2e}", flush=True)

        avg_loss = epoch_loss / num_batches
        print(f"\nEpoch {epoch} complete | Avg Loss: {avg_loss:.4f}", flush=True)

        # Save checkpoint
        lora_state = {name: param.data.cpu() for name, param in model.named_parameters() if param.requires_grad}
        ckpt_path = os.path.join(config.output_dir, f"epoch_{epoch}.pt")
        torch.save({'lora_state_dict': lora_state, 'config': config}, ckpt_path)
        print(f"  Saved {ckpt_path}", flush=True)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(config.output_dir, "best_model.pt")
            torch.save({'lora_state_dict': lora_state, 'config': config}, best_path)
            print(f"  New best model saved!", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("TRAINING COMPLETE", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
