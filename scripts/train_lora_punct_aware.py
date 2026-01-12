#!/usr/bin/env python3
"""
Train FRESH LoRA from V15 base for punct-aware correction.

Key differences from previous training:
1. Fresh LoRA - NOT stacked on hamza checkpoint
2. Punct learned ONLY from QALB (other data has punct stripped)
3. Content learned from all sources

Based on same principle that fixed hamza: learn from correct source only.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
import json
import math
import os
from pathlib import Path
from dataclasses import dataclass
from tqdm import tqdm

@dataclass
class LoRAConfig:
    base_model_path: str = "/home/ubuntu/nahawi/fasih_v15_model/best_model.pt"
    train_data: str = "/home/ubuntu/nahawi/data/punct_aware/train_punct_aware.json"
    dev_data: str = "/home/ubuntu/nahawi/data/qalb_real_dev.json"
    output_dir: str = "/home/ubuntu/nahawi/lora_punct_aware/"
    tokenizer_path: str = "/home/ubuntu/nahawi/nahawi_spm.model"

    # LoRA params
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05

    # Training params - USE THE VRAM!
    batch_size: int = 224
    learning_rate: float = 3e-4  # Slightly higher for larger batch
    num_epochs: int = 3
    warmup_steps: int = 200
    max_seq_len: int = 256
    gradient_accumulation_steps: int = 1
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 50


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
    def __init__(self, original, rank, alpha, dropout=0.0):
        super().__init__()
        self.original = original
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.zeros(rank, original.in_features))
        self.lora_B = nn.Parameter(torch.zeros(original.out_features, rank))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize A with small random values, B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    @property
    def weight(self):
        return self.original.weight

    @property
    def bias(self):
        return self.original.bias

    def forward(self, x):
        base_out = self.original(x)
        lora_out = self.dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling
        return base_out + lora_out


class NahawiGEC(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config["vocab_size"], config["d_model"])
        self.pos_encoder = PositionalEncoding(config["d_model"], config["max_seq_len"])

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config["d_model"],
            nhead=config["nhead"],
            dim_feedforward=config["dim_feedforward"],
            dropout=config["dropout"],
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config["num_encoder_layers"])

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config["d_model"],
            nhead=config["nhead"],
            dim_feedforward=config["dim_feedforward"],
            dropout=config["dropout"],
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config["num_decoder_layers"])

        self.output_projection = nn.Linear(config["d_model"], config["vocab_size"])
        self.output_projection.weight = self.embedding.weight
        self.d_model = config["d_model"]

    def add_lora(self, rank=64, alpha=128, dropout=0.05):
        """Add LoRA to attention and FFN layers."""
        for layer in self.encoder.layers:
            layer.self_attn.out_proj = LoRALinear(layer.self_attn.out_proj, rank, alpha, dropout)
            layer.linear1 = LoRALinear(layer.linear1, rank, alpha, dropout)
            layer.linear2 = LoRALinear(layer.linear2, rank, alpha, dropout)

        for layer in self.decoder.layers:
            layer.self_attn.out_proj = LoRALinear(layer.self_attn.out_proj, rank, alpha, dropout)
            layer.multihead_attn.out_proj = LoRALinear(layer.multihead_attn.out_proj, rank, alpha, dropout)
            layer.linear1 = LoRALinear(layer.linear1, rank, alpha, dropout)
            layer.linear2 = LoRALinear(layer.linear2, rank, alpha, dropout)

    def get_lora_params(self):
        """Get only LoRA parameters for optimization."""
        lora_params = []
        for name, param in self.named_parameters():
            if 'lora_A' in name or 'lora_B' in name:
                lora_params.append(param)
        return lora_params

    def save_lora(self, path):
        """Save only LoRA weights."""
        lora_state = {}
        for name, param in self.named_parameters():
            if 'lora_A' in name or 'lora_B' in name:
                lora_state[name] = param.data.cpu()
        torch.save({'lora_state_dict': lora_state}, path)

    def forward(self, src, tgt):
        src_emb = self.pos_encoder(self.embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.embedding(tgt) * math.sqrt(self.d_model))

        memory = self.encoder(src_emb)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1), device=tgt.device)
        output = self.decoder(tgt_emb, memory, tgt_mask=causal_mask)

        return self.output_projection(output)

    @torch.no_grad()
    def generate(self, src_ids, max_len=256, eos_id=3):
        self.eval()
        device = src_ids.device
        src_emb = self.pos_encoder(self.embedding(src_ids) * math.sqrt(self.d_model))
        memory = self.encoder(src_emb)

        generated = torch.full((src_ids.size(0), 1), 2, dtype=torch.long, device=device)
        for _ in range(max_len - 1):
            tgt_emb = self.pos_encoder(self.embedding(generated) * math.sqrt(self.d_model))
            causal_mask = nn.Transformer.generate_square_subsequent_mask(generated.size(1), device=device)
            output = self.decoder(tgt_emb, memory, tgt_mask=causal_mask)
            next_token = self.output_projection(output[:, -1, :]).argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if (next_token == eos_id).all():
                break
        return generated


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

    src_padded = nn.utils.rnn.pad_sequence(src_ids, batch_first=True, padding_value=0)
    tgt_padded = nn.utils.rnn.pad_sequence(tgt_ids, batch_first=True, padding_value=0)

    return {'src_ids': src_padded, 'tgt_ids': tgt_padded}


PUNCT_SET = set('،.؟!؛:,;?')


def compute_f05(ref, hyp):
    ref_tokens = ref.split()
    hyp_tokens = hyp.split()
    if not ref_tokens:
        return 1.0 if not hyp_tokens else 0.0
    matches = len(set(enumerate(ref_tokens)) & set(enumerate(hyp_tokens)))
    p = matches / len(hyp_tokens) if hyp_tokens else 0
    r = matches / len(ref_tokens) if ref_tokens else 0
    if p + r == 0:
        return 0.0
    return 1.25 * p * r / (0.25 * p + r)


def strip_punct_text(text):
    return ''.join(c for c in text if c not in PUNCT_SET)


def evaluate(model, tokenizer, dev_data, device, num_samples=200):
    """Evaluate on dev set - both with and without punct."""
    model.eval()
    f05_wp = 0  # with punct
    f05_np = 0  # no punct

    samples = dev_data[:num_samples]
    for item in samples:
        src = item.get('source', item.get('src', ''))
        ref = item.get('target', item.get('tgt', ''))

        src_ids = torch.tensor([[2] + tokenizer.encode(src)[:254] + [3]], device=device)
        with torch.no_grad():
            hyp_ids = model.generate(src_ids)
        hyp = tokenizer.decode(hyp_ids[0].tolist()[1:]).replace('</s>', '').strip()

        f05_wp += compute_f05(ref, hyp)
        f05_np += compute_f05(strip_punct_text(ref), strip_punct_text(hyp))

    return {
        'f05_with_punct': 100 * f05_wp / len(samples),
        'f05_no_punct': 100 * f05_np / len(samples)
    }


def main():
    config = LoRAConfig()

    print("=" * 70)
    print("TRAIN FRESH LoRA - PUNCT AWARE")
    print("=" * 70)
    print("\nPrinciple: Punct learned ONLY from QALB")
    print("           Content learned from all sources")
    print()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create output directory
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(config.tokenizer_path)

    # Load base model
    print("Loading V15 base model...")
    model_config = {
        "vocab_size": 32000,
        "d_model": 768,
        "nhead": 12,
        "num_encoder_layers": 6,
        "num_decoder_layers": 6,
        "dim_feedforward": 3072,
        "dropout": 0.1,
        "max_seq_len": 256
    }

    model = NahawiGEC(model_config)
    checkpoint = torch.load(config.base_model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint.get('model_state_dict', checkpoint), strict=False)

    # Add FRESH LoRA (not loading any previous LoRA weights)
    print("Adding FRESH LoRA layers...")
    model.add_lora(rank=config.lora_rank, alpha=config.lora_alpha, dropout=config.lora_dropout)
    model.to(device)

    # Freeze base model, train only LoRA
    for name, param in model.named_parameters():
        if 'lora_A' not in name and 'lora_B' not in name:
            param.requires_grad = False

    lora_params = model.get_lora_params()
    print(f"Trainable LoRA params: {sum(p.numel() for p in lora_params):,}")

    # Load training data
    print(f"\nLoading training data from {config.train_data}...")
    with open(config.train_data, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    print(f"Training examples: {len(train_data):,}")

    # Show data composition
    type_counts = {}
    for item in train_data:
        t = item.get('type', 'unknown')
        type_counts[t] = type_counts.get(t, 0) + 1
    print("Data composition:")
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {t}: {c:,} ({100*c/len(train_data):.1f}%)")

    # Load dev data
    print(f"\nLoading dev data from {config.dev_data}...")
    with open(config.dev_data, 'r', encoding='utf-8') as f:
        dev_data = json.load(f)
    print(f"Dev examples: {len(dev_data)}")

    # Create dataset and dataloader
    train_dataset = GECDataset(train_data, tokenizer, config.max_seq_len)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    # Optimizer and scheduler
    optimizer = optim.AdamW(lora_params, lr=config.learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * config.num_epochs // config.gradient_accumulation_steps
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        total_steps=total_steps,
        pct_start=config.warmup_steps / total_steps
    )

    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # Initial evaluation
    print("\n" + "=" * 50)
    print("INITIAL EVALUATION (V15 base, no LoRA training)")
    print("=" * 50)
    init_metrics = evaluate(model, tokenizer, dev_data, device)
    print(f"  F0.5 with punct:    {init_metrics['f05_with_punct']:.2f}%")
    print(f"  F0.5 without punct: {init_metrics['f05_no_punct']:.2f}%")

    # Training loop
    print("\n" + "=" * 50)
    print("TRAINING")
    print("=" * 50)

    best_f05_wp = init_metrics['f05_with_punct']
    global_step = 0

    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        for batch_idx, batch in enumerate(pbar):
            src_ids = batch['src_ids'].to(device)
            tgt_ids = batch['tgt_ids'].to(device)

            # Forward pass
            logits = model(src_ids, tgt_ids[:, :-1])
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_ids[:, 1:].reshape(-1))
            loss = loss / config.gradient_accumulation_steps

            loss.backward()
            epoch_loss += loss.item() * config.gradient_accumulation_steps

            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Logging
                if global_step % config.logging_steps == 0:
                    avg_loss = epoch_loss / (batch_idx + 1)
                    pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'lr': f'{scheduler.get_last_lr()[0]:.2e}'})

                # Evaluation
                if global_step % config.eval_steps == 0:
                    metrics = evaluate(model, tokenizer, dev_data, device)
                    print(f"\n  [Step {global_step}] F0.5 wp={metrics['f05_with_punct']:.2f}% np={metrics['f05_no_punct']:.2f}%")

                    if metrics['f05_with_punct'] > best_f05_wp:
                        best_f05_wp = metrics['f05_with_punct']
                        model.save_lora(os.path.join(config.output_dir, 'best_lora.pt'))
                        print(f"  -> New best! Saved.")

                    model.train()

                # Save checkpoint
                if global_step % config.save_steps == 0:
                    model.save_lora(os.path.join(config.output_dir, f'step_{global_step}.pt'))

        # End of epoch evaluation
        print(f"\n{'='*50}")
        print(f"END OF EPOCH {epoch+1}")
        print(f"{'='*50}")

        metrics = evaluate(model, tokenizer, dev_data, device, num_samples=500)
        print(f"  F0.5 with punct:    {metrics['f05_with_punct']:.2f}%")
        print(f"  F0.5 without punct: {metrics['f05_no_punct']:.2f}%")

        # Save epoch checkpoint
        model.save_lora(os.path.join(config.output_dir, f'epoch_{epoch+1}.pt'))

        if metrics['f05_with_punct'] > best_f05_wp:
            best_f05_wp = metrics['f05_with_punct']
            model.save_lora(os.path.join(config.output_dir, 'best_lora.pt'))
            print(f"  -> New best!")

    # Final evaluation on full dev set
    print("\n" + "=" * 50)
    print("FINAL EVALUATION (full dev set)")
    print("=" * 50)

    final_metrics = evaluate(model, tokenizer, dev_data, device, num_samples=len(dev_data))
    print(f"  F0.5 with punct:    {final_metrics['f05_with_punct']:.2f}%")
    print(f"  F0.5 without punct: {final_metrics['f05_no_punct']:.2f}%")
    print(f"\n  Best F0.5 with punct: {best_f05_wp:.2f}%")
    print(f"  Improvement from baseline (71.34%): {best_f05_wp - 71.34:+.2f}%")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print(f"Output: {config.output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
