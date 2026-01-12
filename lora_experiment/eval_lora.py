#!/usr/bin/env python3
"""
Evaluate LoRA model on QALB dev set.
Reports F0.5 with and without punct, plus FASIH error breakdown.
"""

import torch
import torch.nn as nn
import sentencepiece as spm
import json
import os
import math
from collections import defaultdict
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class LoRAConfig:
    """LoRA configuration - needed for checkpoint loading."""
    base_model_path: str = ""
    tokenizer_path: str = ""
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    train_data: str = ""
    output_dir: str = ""
    batch_size: int = 192
    gradient_accumulation: int = 4
    learning_rate: float = 2e-4
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    max_seq_len: int = 256
    eval_every: int = 1000
    save_every: int = 2000


# =============================================================================
# CONFIG
# =============================================================================

LORA_MODEL_PATH = "/home/ubuntu/nahawi/lora_model_v2/best_model.pt"
BASE_MODEL_PATH = "/home/ubuntu/nahawi/gec_clean/epoch_11.pt"
TOKENIZER_PATH = "/home/ubuntu/nahawi/nahawi_spm.model"
DEV_DATA = "/home/ubuntu/nahawi/data/fasih_test.json"
PUNCT_CLASSIFIER_PATH = "/home/ubuntu/nahawi/punct_classifier/best_model.pt"


# =============================================================================
# MODEL (same as train_lora.py)
# =============================================================================

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
        self.alpha = alpha
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


class NahawiGECWithLoRA(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config

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

    def add_lora_adapters(self, rank=32, alpha=64):
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


# =============================================================================
# ERROR TYPE DETECTION
# =============================================================================

def classify_error(src_word: str, tgt_word: str) -> str:
    """Classify the error type for a word-level correction."""
    if src_word == tgt_word:
        return 'none'

    # Hamza
    hamza_chars = set('أإآءؤئ')
    if (hamza_chars & set(src_word)) != (hamza_chars & set(tgt_word)):
        return 'hamza'

    # Taa marbuta
    if ('ة' in src_word) != ('ة' in tgt_word) or ('ه' in src_word) != ('ه' in tgt_word):
        return 'taa_marbuta'

    # Alif maqsura
    if ('ى' in src_word) != ('ى' in tgt_word) or ('ي' in src_word) != ('ي' in tgt_word):
        return 'alif_maqsura'

    # Letter confusion
    confusion_pairs = [('ض', 'ظ'), ('د', 'ذ'), ('ت', 'ط'), ('س', 'ص')]
    for c1, c2 in confusion_pairs:
        if (c1 in src_word and c2 in tgt_word) or (c2 in src_word and c1 in tgt_word):
            return 'letter_confusion'

    # Spacing
    if ' ' in src_word or ' ' in tgt_word:
        return 'spacing'

    return 'other'


def compute_f05(tp: int, fp: int, fn: int) -> float:
    """Compute F0.5 score."""
    if tp + fp == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)

    if tp + fn == 0:
        recall = 0
    else:
        recall = tp / (tp + fn)

    if precision + recall == 0:
        return 0

    beta = 0.5
    f05 = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)
    return f05


def strip_punct(text: str) -> str:
    """Remove punctuation from text."""
    punct = set('،؛؟!.,:;?')
    return ''.join(c for c in text if c not in punct)


def evaluate_pair(hyp: str, ref: str, src: str, include_punct: bool = True) -> Dict:
    """
    Evaluate a single hypothesis against reference.
    Returns TP, FP, FN counts.
    """
    if not include_punct:
        hyp = strip_punct(hyp)
        ref = strip_punct(ref)
        src = strip_punct(src)

    # Word-level comparison
    hyp_words = hyp.split()
    ref_words = ref.split()
    src_words = src.split()

    # Find edits
    tp = fp = fn = 0

    # Build edit sets
    ref_edits = set()
    hyp_edits = set()

    for i, (s, r) in enumerate(zip(src_words, ref_words)):
        if s != r:
            ref_edits.add((i, s, r))

    for i, (s, h) in enumerate(zip(src_words, hyp_words)):
        if s != h:
            hyp_edits.add((i, s, h))

    # Count matches
    for edit in hyp_edits:
        if edit in ref_edits:
            tp += 1
        else:
            fp += 1

    for edit in ref_edits:
        if edit not in hyp_edits:
            fn += 1

    return {'tp': tp, 'fp': fp, 'fn': fn}


# =============================================================================
# MAIN EVALUATION
# =============================================================================

def load_model(base_path: str, lora_path: str, device: torch.device):
    """Load base model + LoRA weights."""
    config = {
        "vocab_size": 32000,
        "d_model": 768,
        "nhead": 12,
        "num_encoder_layers": 6,
        "num_decoder_layers": 6,
        "dim_feedforward": 3072,
        "dropout": 0.1,
        "max_seq_len": 256,
    }

    model = NahawiGECWithLoRA(config)

    # Load base
    base_ckpt = torch.load(base_path, map_location='cpu', weights_only=False)
    state_dict = base_ckpt.get('model_state_dict', base_ckpt)
    model.load_state_dict(state_dict, strict=False)

    # Add LoRA structure
    model.add_lora_adapters()

    # Load LoRA weights
    if os.path.exists(lora_path):
        lora_ckpt = torch.load(lora_path, map_location='cpu', weights_only=False)
        lora_state = lora_ckpt.get('lora_state_dict', {})
        model.load_state_dict(lora_state, strict=False)
        print(f"Loaded LoRA weights from {lora_path}")
    else:
        print(f"Warning: LoRA path not found: {lora_path}")

    model.to(device)
    model.eval()
    return model


def main():
    print("=" * 60)
    print("LORA MODEL EVALUATION")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(TOKENIZER_PATH)

    # Load model
    print("\nLoading model...")
    model = load_model(BASE_MODEL_PATH, LORA_MODEL_PATH, device)

    # Load dev data
    print(f"\nLoading dev data from {DEV_DATA}...")
    with open(DEV_DATA, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        if content.startswith('['):
            dev_data = json.loads(content)
        else:
            dev_data = [json.loads(line) for line in content.split('\n') if line.strip()]

    print(f"  Loaded {len(dev_data)} examples")

    # Evaluate
    print("\nGenerating hypotheses...")
    results_with_punct = {'tp': 0, 'fp': 0, 'fn': 0}
    results_no_punct = {'tp': 0, 'fp': 0, 'fn': 0}
    error_type_results = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})

    for i, item in enumerate(dev_data):
        src = item.get('source', item.get('src', ''))
        ref = item.get('target', item.get('tgt', ''))

        if not src or not ref:
            continue

        # Generate
        src_ids = [2] + tokenizer.encode(src)[:254] + [3]
        src_tensor = torch.tensor([src_ids], device=device)
        hyp_ids = model.generate(src_tensor, max_len=256)
        hyp = tokenizer.decode(hyp_ids[0].tolist()[1:])  # Skip BOS
        hyp = hyp.replace('</s>', '').strip()

        # Evaluate with punct
        r = evaluate_pair(hyp, ref, src, include_punct=True)
        results_with_punct['tp'] += r['tp']
        results_with_punct['fp'] += r['fp']
        results_with_punct['fn'] += r['fn']

        # Evaluate without punct
        r = evaluate_pair(hyp, ref, src, include_punct=False)
        results_no_punct['tp'] += r['tp']
        results_no_punct['fp'] += r['fp']
        results_no_punct['fn'] += r['fn']

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(dev_data)}")

    # Compute F0.5
    f05_with_punct = compute_f05(
        results_with_punct['tp'],
        results_with_punct['fp'],
        results_with_punct['fn']
    )
    f05_no_punct = compute_f05(
        results_no_punct['tp'],
        results_no_punct['fp'],
        results_no_punct['fn']
    )

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nF0.5 (with punct):    {f05_with_punct*100:.2f}%")
    print(f"F0.5 (without punct): {f05_no_punct*100:.2f}%")

    print(f"\nWith punct:")
    print(f"  TP: {results_with_punct['tp']}")
    print(f"  FP: {results_with_punct['fp']}")
    print(f"  FN: {results_with_punct['fn']}")

    print(f"\nWithout punct:")
    print(f"  TP: {results_no_punct['tp']}")
    print(f"  FP: {results_no_punct['fp']}")
    print(f"  FN: {results_no_punct['fn']}")

    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"Base model (epoch 11):  81.05% (no punct) / 55.32% (with punct filter)")
    print(f"LoRA model:             {f05_no_punct*100:.2f}% (no punct) / {f05_with_punct*100:.2f}% (with punct)")
    print(f"SOTA (ArbESC+):         82.63%")

    if f05_no_punct > 0.8263:
        print("\n🎉 BEAT SOTA! Data distribution was the bottleneck.")
    elif f05_no_punct >= 0.81:
        print("\n📊 Matched base. Capacity might be the constraint.")
    else:
        print("\n⚠️ Below base. Something went wrong.")


if __name__ == "__main__":
    main()
