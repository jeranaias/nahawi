#!/usr/bin/env python3
"""
Two-pass correction test: Run model twice, see if second pass catches residuals.
"""

import torch
import torch.nn as nn
import sentencepiece as spm
import json
import math
from dataclasses import dataclass


@dataclass
class LoRAConfig:
    base_model_path: str = ""
    epoch1_lora_path: str = ""
    tokenizer_path: str = ""
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    train_data: str = ""
    output_dir: str = ""
    batch_size: int = 192
    gradient_accumulation: int = 4
    learning_rate: float = 1e-4
    num_epochs: int = 2
    start_epoch: int = 2
    max_seq_len: int = 256


BASE_MODEL_PATH = "/home/ubuntu/nahawi/fasih_v15_model/best_model.pt"
LORA_PATH = "/home/ubuntu/nahawi/lora_model_hamza/epoch_1.pt"
TOKENIZER_PATH = "/home/ubuntu/nahawi/nahawi_spm.model"
FASIH_TEST = "/home/ubuntu/nahawi/data/fasih_test.json"

PUNCT_SET = set("،؛؟!.,:;?")


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


def strip_punct(text):
    return ''.join(c for c in text if c not in PUNCT_SET)


def compute_f05(ref, hyp, ignore_punct=False):
    if ignore_punct:
        ref, hyp = strip_punct(ref), strip_punct(hyp)
    ref_tokens, hyp_tokens = ref.split(), hyp.split()
    if not ref_tokens:
        return 1.0 if not hyp_tokens else 0.0
    matches = len(set(enumerate(ref_tokens)) & set(enumerate(hyp_tokens)))
    p = matches / len(hyp_tokens) if hyp_tokens else 0
    r = matches / len(ref_tokens) if ref_tokens else 0
    if p + r == 0:
        return 0.0
    return 1.25 * p * r / (0.25 * p + r)


def main():
    print("=" * 70, flush=True)
    print("TWO-PASS CORRECTION TEST", flush=True)
    print("=" * 70, flush=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)

    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(TOKENIZER_PATH)

    config = {"vocab_size": 32000, "d_model": 768, "nhead": 12, "num_encoder_layers": 6,
              "num_decoder_layers": 6, "dim_feedforward": 3072, "dropout": 0.1, "max_seq_len": 256}

    print("\nLoading model...", flush=True)
    model = NahawiGEC(config)
    base_ckpt = torch.load(BASE_MODEL_PATH, map_location='cpu', weights_only=False)
    model.load_state_dict(base_ckpt.get('model_state_dict', base_ckpt), strict=False)
    model.add_lora()
    lora_ckpt = torch.load(LORA_PATH, map_location='cpu', weights_only=False)
    model.load_state_dict(lora_ckpt.get('lora_state_dict', {}), strict=False)
    model.to(device).eval()
    print("  Model ready", flush=True)

    with open(FASIH_TEST, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"\nEvaluating {len(data)} examples...", flush=True)

    one_pass_f05 = 0
    two_pass_f05 = 0
    improved_count = 0

    for i, item in enumerate(data):
        src = item.get('source', item.get('src', ''))
        ref = item.get('target', item.get('tgt', ''))

        # Pass 1
        src_ids = torch.tensor([[2] + tokenizer.encode(src)[:254] + [3]], device=device)
        hyp1_ids = model.generate(src_ids)
        hyp1 = tokenizer.decode(hyp1_ids[0].tolist()[1:]).replace('</s>', '').strip()

        # Pass 2 (use pass 1 output as input)
        pass2_ids = torch.tensor([[2] + tokenizer.encode(hyp1)[:254] + [3]], device=device)
        hyp2_ids = model.generate(pass2_ids)
        hyp2 = tokenizer.decode(hyp2_ids[0].tolist()[1:]).replace('</s>', '').strip()

        f05_1 = compute_f05(ref, hyp1, ignore_punct=True)
        f05_2 = compute_f05(ref, hyp2, ignore_punct=True)

        one_pass_f05 += f05_1
        two_pass_f05 += f05_2

        if f05_2 > f05_1:
            improved_count += 1

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(data)}] 1-pass: {100*one_pass_f05/(i+1):.2f}% | 2-pass: {100*two_pass_f05/(i+1):.2f}%", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("RESULTS", flush=True)
    print("=" * 70, flush=True)
    print(f"1-pass F0.5: {100*one_pass_f05/len(data):.2f}%", flush=True)
    print(f"2-pass F0.5: {100*two_pass_f05/len(data):.2f}%", flush=True)
    print(f"Improved examples: {improved_count}/{len(data)} ({100*improved_count/len(data):.1f}%)", flush=True)
    print(f"Delta: {100*(two_pass_f05-one_pass_f05)/len(data):+.2f}%", flush=True)


if __name__ == "__main__":
    main()
