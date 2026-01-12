#!/usr/bin/env python3
"""
Evaluate V15 + Hamza LoRA model on QALB and FASIH.
Tests both overall F0.5 and hamza-specific accuracy.
"""

import torch
import torch.nn as nn
import sentencepiece as spm
import json
import math
import re
from dataclasses import dataclass
from collections import defaultdict

# Config
BASE_MODEL_PATH = "/home/ubuntu/nahawi/fasih_v15_model/best_model.pt"
HAMZA_LORA_PATH = "/home/ubuntu/nahawi/lora_model_hamza/epoch_3.pt"
TOKENIZER_PATH = "/home/ubuntu/nahawi/nahawi_spm.model"
QALB_DEV = "/home/ubuntu/nahawi/data/qalb_real_dev.json"
FASIH_TEST = "/home/ubuntu/nahawi/data/fasih_test.json"

PUNCT_CHARS = "،؛؟!.,:;?"
PUNCT_SET = set(PUNCT_CHARS)

@dataclass
class LoRAConfig:
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05


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


class LoRALinear(nn.Module):
    def __init__(self, original_layer: nn.Linear, rank: int, alpha: int, dropout: float = 0.0):
        super().__init__()
        self.original = original_layer
        self.rank = rank
        self.scaling = alpha / rank
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.lora_dropout = nn.Identity()

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


class NahawiGECWithLoRA(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
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

    def add_lora_adapters(self, rank=64, alpha=128):
        for layer in self.encoder.layers:
            layer.self_attn.out_proj = LoRALinear(layer.self_attn.out_proj, rank, alpha)
            layer.linear1 = LoRALinear(layer.linear1, rank, alpha)
            layer.linear2 = LoRALinear(layer.linear2, rank, alpha)
        for layer in self.decoder.layers:
            layer.self_attn.out_proj = LoRALinear(layer.self_attn.out_proj, rank, alpha)
            layer.multihead_attn.out_proj = LoRALinear(layer.multihead_attn.out_proj, rank, alpha)
            layer.linear1 = LoRALinear(layer.linear1, rank, alpha)
            layer.linear2 = LoRALinear(layer.linear2, rank, alpha)

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


def strip_punct(text):
    return ''.join(c for c in text if c not in PUNCT_SET)


def compute_f05(ref, hyp, ignore_punct=False):
    if ignore_punct:
        ref = strip_punct(ref)
        hyp = strip_punct(hyp)
    ref_tokens = ref.split()
    hyp_tokens = hyp.split()
    if not ref_tokens:
        return 1.0 if not hyp_tokens else 0.0
    ref_set = set(enumerate(ref_tokens))
    hyp_set = set(enumerate(hyp_tokens))
    matches = len(ref_set & hyp_set)
    precision = matches / len(hyp_tokens) if hyp_tokens else 0
    recall = matches / len(ref_tokens) if ref_tokens else 0
    if precision + recall == 0:
        return 0.0
    beta = 0.5
    f05 = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)
    return f05


def classify_hamza_error(src_word, tgt_word):
    """Check if this is a hamza-related error."""
    hamza_chars = 'إأآؤئء'

    # Check if target has hamza that source is missing
    if any(c in tgt_word for c in hamza_chars) and not any(c in src_word for c in hamza_chars):
        if 'إ' in tgt_word:
            return 'hamza_add_alif_kasra'
        elif 'أ' in tgt_word:
            return 'hamza_add_alif'
        elif 'آ' in tgt_word:
            return 'alif_madda'
        elif 'ؤ' in tgt_word:
            return 'hamza_waw'
        elif 'ئ' in tgt_word:
            return 'hamza_yaa'
        return 'hamza_other'

    # Check for hamza position swap
    if set(src_word) & set(hamza_chars) and set(tgt_word) & set(hamza_chars):
        return 'hamza_swap'

    return None


def evaluate_dataset(model, tokenizer, device, data_path, name):
    """Evaluate model on dataset."""
    print(f"\n{'='*70}", flush=True)
    print(f"EVALUATING: {name}", flush=True)
    print(f"{'='*70}", flush=True)

    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} examples", flush=True)

    f05_no_punct_sum = 0
    f05_with_punct_sum = 0

    # Hamza tracking
    hamza_stats = defaultdict(lambda: {'total': 0, 'correct': 0})

    for i, item in enumerate(data):
        src = item.get('source', item.get('src', ''))
        ref = item.get('target', item.get('tgt', ''))

        # Generate
        src_ids = [2] + tokenizer.encode(src)[:254] + [3]
        src_tensor = torch.tensor([src_ids], device=device)
        hyp_ids = model.generate(src_tensor, max_len=256)
        hyp = tokenizer.decode(hyp_ids[0].tolist()[1:]).replace('</s>', '').strip()

        # Compute F0.5
        f05_no_punct = compute_f05(ref, hyp, ignore_punct=True)
        f05_with_punct = compute_f05(ref, hyp, ignore_punct=False)
        f05_no_punct_sum += f05_no_punct
        f05_with_punct_sum += f05_with_punct

        # Track hamza errors
        src_words = src.split()
        ref_words = ref.split()
        hyp_words = hyp.split()

        for sw, rw in zip(src_words, ref_words):
            error_type = classify_hamza_error(sw, rw)
            if error_type:
                hamza_stats[error_type]['total'] += 1
                # Check if model got it right
                if rw in hyp:
                    hamza_stats[error_type]['correct'] += 1

        if (i + 1) % 100 == 0:
            avg_f05 = 100 * f05_no_punct_sum / (i + 1)
            print(f"  [{i+1}/{len(data)}] Running F0.5 (no punct): {avg_f05:.2f}%", flush=True)

    # Final results
    avg_f05_no_punct = 100 * f05_no_punct_sum / len(data)
    avg_f05_with_punct = 100 * f05_with_punct_sum / len(data)

    print(f"\n--- {name} RESULTS ---", flush=True)
    print(f"F0.5 (no punct):   {avg_f05_no_punct:.2f}%", flush=True)
    print(f"F0.5 (with punct): {avg_f05_with_punct:.2f}%", flush=True)

    print(f"\n--- HAMZA ACCURACY ---", flush=True)
    for error_type in sorted(hamza_stats.keys()):
        stats = hamza_stats[error_type]
        if stats['total'] > 0:
            acc = 100 * stats['correct'] / stats['total']
            print(f"  {error_type}: {stats['correct']}/{stats['total']} ({acc:.1f}%)", flush=True)

    return avg_f05_no_punct, avg_f05_with_punct, hamza_stats


def main():
    print("=" * 70, flush=True)
    print("V15 + HAMZA LORA EVALUATION", flush=True)
    print("=" * 70, flush=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)

    # Load tokenizer
    print("\nLoading tokenizer...", flush=True)
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(TOKENIZER_PATH)
    print("  Tokenizer loaded", flush=True)

    # Model config
    model_config = {
        "vocab_size": 32000, "d_model": 768, "nhead": 12,
        "num_encoder_layers": 6, "num_decoder_layers": 6,
        "dim_feedforward": 3072, "dropout": 0.1, "max_seq_len": 256,
    }

    # Load model
    print("\nLoading V15 + Hamza LoRA model...", flush=True)
    model = NahawiGECWithLoRA(model_config)

    # Load base weights
    base_ckpt = torch.load(BASE_MODEL_PATH, map_location='cpu', weights_only=False)
    state_dict = base_ckpt.get('model_state_dict', base_ckpt)
    model.load_state_dict(state_dict, strict=False)
    print("  Base V15 loaded", flush=True)

    # Add LoRA and load weights
    model.add_lora_adapters(rank=64, alpha=128)
    lora_ckpt = torch.load(HAMZA_LORA_PATH, map_location='cpu', weights_only=False)
    model.load_state_dict(lora_ckpt.get('lora_state_dict', {}), strict=False)
    print("  Hamza LoRA weights loaded", flush=True)

    model.to(device).eval()
    print("  Model ready", flush=True)

    # Evaluate on both datasets
    qalb_results = evaluate_dataset(model, tokenizer, device, QALB_DEV, "QALB Dev")
    fasih_results = evaluate_dataset(model, tokenizer, device, FASIH_TEST, "FASIH Test")

    # Summary
    print("\n" + "=" * 70, flush=True)
    print("FINAL SUMMARY", flush=True)
    print("=" * 70, flush=True)

    print(f"\nQALB Dev:", flush=True)
    print(f"  No punct:   {qalb_results[0]:.2f}%", flush=True)
    print(f"  With punct: {qalb_results[1]:.2f}%", flush=True)

    print(f"\nFASIH Test:", flush=True)
    print(f"  No punct:   {fasih_results[0]:.2f}%", flush=True)
    print(f"  With punct: {fasih_results[1]:.2f}%", flush=True)

    print(f"\nPrevious V15+LoRA (rank 32):", flush=True)
    print(f"  QALB:  81.29% no-punct", flush=True)
    print(f"  FASIH: 84.72% no-punct", flush=True)

    print(f"\nSOTA (ArbESC+): 82.63%", flush=True)


if __name__ == "__main__":
    main()
