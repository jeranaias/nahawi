#!/usr/bin/env python3
"""
Error analysis on FASIH using FIXED classifier.
Categorizes by WHAT CHANGED, not by what else is in the word.
"""

import torch
import torch.nn as nn
import sentencepiece as spm
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from classify_error_fixed import classify_error_fixed

BASE_MODEL_PATH = "/home/ubuntu/nahawi/fasih_v15_model/best_model.pt"
LORA_PATH = "/home/ubuntu/nahawi/lora_model_hamza/epoch_1.pt"
TOKENIZER_PATH = "/home/ubuntu/nahawi/nahawi_spm.model"
FASIH_TEST = "/home/ubuntu/nahawi/data/fasih_test.json"

@dataclass
class LoRAConfig:
    base_model_path: str = ""
    lora_rank: int = 64
    lora_alpha: int = 128

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
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.zeros(rank, original.in_features))
        self.lora_B = nn.Parameter(torch.zeros(original.out_features, rank))
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
        encoder_layer = nn.TransformerEncoderLayer(d_model=config["d_model"], nhead=config["nhead"],
            dim_feedforward=config["dim_feedforward"], dropout=config["dropout"], batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config["num_encoder_layers"])
        decoder_layer = nn.TransformerDecoderLayer(d_model=config["d_model"], nhead=config["nhead"],
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


def main():
    print("=" * 80)
    print("FASIH ERROR ANALYSIS WITH FIXED CLASSIFIER")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load tokenizer
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(TOKENIZER_PATH)

    # Load model
    config = {"vocab_size": 32000, "d_model": 768, "nhead": 12, "num_encoder_layers": 6,
              "num_decoder_layers": 6, "dim_feedforward": 3072, "dropout": 0.1, "max_seq_len": 256}

    model = NahawiGEC(config)
    base_ckpt = torch.load(BASE_MODEL_PATH, map_location='cpu', weights_only=False)
    model.load_state_dict(base_ckpt.get('model_state_dict', base_ckpt), strict=False)
    model.add_lora()
    lora_ckpt = torch.load(LORA_PATH, map_location='cpu', weights_only=False)
    model.load_state_dict(lora_ckpt.get('lora_state_dict', {}), strict=False)
    model.to(device).eval()
    print("Model loaded (V15 + Hamza LoRA epoch 1)")

    # Load FASIH test
    with open(FASIH_TEST, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} FASIH test examples")

    # Track errors by type
    error_stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'examples': []})

    for i, item in enumerate(data):
        src = item.get('source', item.get('src', ''))
        ref = item.get('target', item.get('tgt', ''))

        # Generate hypothesis
        src_ids = torch.tensor([[2] + tokenizer.encode(src)[:254] + [3]], device=device)
        hyp_ids = model.generate(src_ids)
        hyp = tokenizer.decode(hyp_ids[0].tolist()[1:]).replace('</s>', '').strip()

        # Align words and check errors
        src_words = src.split()
        ref_words = ref.split()
        hyp_words = hyp.split()

        for j, (sw, rw) in enumerate(zip(src_words, ref_words)):
            if sw != rw:
                error_type, desc = classify_error_fixed(sw, rw)
                if error_type:
                    error_stats[error_type]['total'] += 1

                    # Check if model got it right (word appears in hyp at similar position)
                    got_correct = False
                    if j < len(hyp_words) and hyp_words[j] == rw:
                        got_correct = True
                    elif rw in hyp_words:
                        # Check within ±2 positions
                        for k in range(max(0, j-2), min(len(hyp_words), j+3)):
                            if hyp_words[k] == rw:
                                got_correct = True
                                break

                    if got_correct:
                        error_stats[error_type]['correct'] += 1
                    elif len(error_stats[error_type]['examples']) < 3:
                        # Store first 3 failures as examples
                        error_stats[error_type]['examples'].append({
                            'src': sw, 'ref': rw,
                            'hyp': hyp_words[j] if j < len(hyp_words) else '<missing>'
                        })

        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{len(data)}] processed", flush=True)

    # Print results
    print("\n" + "=" * 80)
    print("ERROR BREAKDOWN BY TYPE (FASIH)")
    print("=" * 80)

    # Sort by total count descending
    sorted_types = sorted(error_stats.items(), key=lambda x: x[1]['total'], reverse=True)

    total_errors = sum(s['total'] for _, s in sorted_types)
    total_correct = sum(s['correct'] for _, s in sorted_types)

    print(f"\n{'Error Type':<25} {'Total':>8} {'Correct':>8} {'Accuracy':>10} {'% of All':>10}")
    print("-" * 65)

    for error_type, stats in sorted_types:
        total = stats['total']
        correct = stats['correct']
        acc = 100 * correct / total if total > 0 else 0
        pct = 100 * total / total_errors if total_errors > 0 else 0

        marker = "⚠️" if acc < 50 else "✓" if acc >= 80 else ""
        print(f"{error_type:<25} {total:>8} {correct:>8} {acc:>9.1f}% {pct:>9.1f}% {marker}")

    print("-" * 65)
    overall_acc = 100 * total_correct / total_errors if total_errors > 0 else 0
    print(f"{'TOTAL':<25} {total_errors:>8} {total_correct:>8} {overall_acc:>9.1f}%")

    # Show examples of failures for low-accuracy types
    print("\n" + "=" * 80)
    print("FAILURE EXAMPLES (for types with <70% accuracy)")
    print("=" * 80)

    for error_type, stats in sorted_types:
        total = stats['total']
        correct = stats['correct']
        acc = 100 * correct / total if total > 0 else 0

        if acc < 70 and stats['examples']:
            print(f"\n{error_type} ({acc:.1f}% accuracy):")
            for ex in stats['examples']:
                print(f"  src: {ex['src']} → ref: {ex['ref']} | got: {ex['hyp']}")


if __name__ == "__main__":
    main()
