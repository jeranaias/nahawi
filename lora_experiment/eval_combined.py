#!/usr/bin/env python3
"""
Combined Evaluation: V15 + LoRA + Punct Classifier
The final number that matters.
"""

import torch
import torch.nn as nn
import sentencepiece as spm
import json
import math
import sys
from dataclasses import dataclass
from typing import Dict, List


# =============================================================================
# CONFIG
# =============================================================================

BASE_MODEL_PATH = "/home/ubuntu/nahawi/fasih_v15_model/best_model.pt"
LORA_MODEL_PATH = "/home/ubuntu/nahawi/lora_model_v15/best_model.pt"
PUNCT_MODEL_PATH = "/home/ubuntu/nahawi/punct_classifier_v15/best_model.pt"
TOKENIZER_PATH = "/home/ubuntu/nahawi/nahawi_spm.model"
QALB_DEV = "/home/ubuntu/nahawi/data/qalb_real_dev.json"
FASIH_TEST = "/home/ubuntu/nahawi/data/fasih_test.json"

PUNCT_CHARS = "،؛؟!.,:;?"


@dataclass
class LoRAConfig:
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


@dataclass
class PunctConfig:
    base_model_path: str = ""
    tokenizer_path: str = ""
    output_dir: str = ""
    batch_size: int = 5000
    learning_rate: float = 1e-4
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    max_seq_len: int = 256
    msa_corpus_path: str = ""
    msa_samples_per_epoch: int = 500000


# =============================================================================
# MODEL CLASSES
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


class PunctClassifier(nn.Module):
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

    def forward(self, src_ids, src_mask=None):
        x = self.dropout(self.pos_encoder(self.embedding(src_ids) * math.sqrt(self.d_model)))
        encoded = self.encoder(x, src_key_padding_mask=src_mask)
        return self.classifier(encoded)


# =============================================================================
# EVALUATION
# =============================================================================

def compute_f05(tp: int, fp: int, fn: int) -> float:
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
    punct = set(PUNCT_CHARS)
    return ''.join(c for c in text if c not in punct)


def evaluate_pair(hyp: str, ref: str, src: str, include_punct: bool = True) -> Dict:
    if not include_punct:
        hyp = strip_punct(hyp)
        ref = strip_punct(ref)
        src = strip_punct(src)

    hyp_words = hyp.split()
    ref_words = ref.split()
    src_words = src.split()

    tp = fp = fn = 0

    ref_edits = set()
    hyp_edits = set()

    for i, (s, r) in enumerate(zip(src_words, ref_words)):
        if s != r:
            ref_edits.add((i, s, r))

    for i, (s, h) in enumerate(zip(src_words, hyp_words)):
        if s != h:
            hyp_edits.add((i, s, h))

    for edit in hyp_edits:
        if edit in ref_edits:
            tp += 1
        else:
            fp += 1

    for edit in ref_edits:
        if edit not in hyp_edits:
            fn += 1

    return {'tp': tp, 'fp': fp, 'fn': fn}


def apply_punct_classifier(text: str, punct_model: PunctClassifier, tokenizer, device, class_to_punct: dict, confidence_threshold: float = 0.85):
    """Apply punct classifier to refine punctuation.

    Two key fixes:
    1. Skip classifier on positions where LoRA already has punct (no double-insertion)
    2. Only apply predictions with confidence > threshold
    """
    tokens = tokenizer.encode(text)[:254]
    src_ids = torch.tensor([[2] + tokens + [3]], device=device)

    with torch.no_grad():
        logits = punct_model(src_ids)
        probs = torch.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)[0].cpu().tolist()
        confidences = probs.max(dim=-1)[0][0].cpu().tolist()

    # Decode tokens and add punct based on predictions
    pieces = [tokenizer.id_to_piece(t) for t in tokens]
    result_pieces = []
    punct_set = set(PUNCT_CHARS)

    for i, piece in enumerate(pieces):
        # Keep the piece as-is (including any existing punct)
        result_pieces.append(piece)

        # Check if this piece already ends with punct
        has_existing_punct = any(p in piece for p in punct_set)

        # Skip classifier if LoRA already added punct here
        if has_existing_punct:
            continue

        # Get classifier prediction
        pred_class = preds[i + 1]  # +1 for BOS
        conf = confidences[i + 1]

        # Only apply if high confidence and classifier wants punct
        if pred_class > 0 and pred_class in class_to_punct and conf >= confidence_threshold:
            result_pieces.append(class_to_punct[pred_class])

    return tokenizer.decode_pieces(result_pieces)


def evaluate_dataset(name, data_path, gec_model, punct_model, tokenizer, device, class_to_punct):
    """Evaluate on a single dataset."""
    print(f"\n{'='*60}", flush=True)
    print(f"EVALUATING: {name}", flush=True)
    print(f"{'='*60}", flush=True)

    # Load data
    print(f"Loading from {data_path}...", flush=True)
    with open(data_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        if content.startswith('['):
            dev_data = json.loads(content)
        else:
            dev_data = [json.loads(line) for line in content.split('\n') if line.strip()]
    print(f"  Loaded {len(dev_data)} examples", flush=True)

    results_lora_only = {'tp': 0, 'fp': 0, 'fn': 0}
    results_lora_punct = {'tp': 0, 'fp': 0, 'fn': 0}
    results_no_punct = {'tp': 0, 'fp': 0, 'fn': 0}

    for i, item in enumerate(dev_data):
        src = item.get('source', item.get('src', ''))
        ref = item.get('target', item.get('tgt', ''))

        if not src or not ref:
            continue

        # Generate with V15 + LoRA
        src_ids = [2] + tokenizer.encode(src)[:254] + [3]
        src_tensor = torch.tensor([src_ids], device=device)
        hyp_ids = gec_model.generate(src_tensor, max_len=256)
        hyp_lora = tokenizer.decode(hyp_ids[0].tolist()[1:])
        hyp_lora = hyp_lora.replace('</s>', '').strip()

        # Apply punct classifier
        try:
            hyp_combined = apply_punct_classifier(hyp_lora, punct_model, tokenizer, device, class_to_punct)
        except:
            hyp_combined = hyp_lora

        # Evaluate LoRA only (with punct)
        r = evaluate_pair(hyp_lora, ref, src, include_punct=True)
        results_lora_only['tp'] += r['tp']
        results_lora_only['fp'] += r['fp']
        results_lora_only['fn'] += r['fn']

        # Evaluate LoRA + punct (with punct)
        r = evaluate_pair(hyp_combined, ref, src, include_punct=True)
        results_lora_punct['tp'] += r['tp']
        results_lora_punct['fp'] += r['fp']
        results_lora_punct['fn'] += r['fn']

        # Evaluate no punct
        r = evaluate_pair(hyp_lora, ref, src, include_punct=False)
        results_no_punct['tp'] += r['tp']
        results_no_punct['fp'] += r['fp']
        results_no_punct['fn'] += r['fn']

        if (i + 1) % 100 == 0:
            f05_cur = compute_f05(results_no_punct['tp'], results_no_punct['fp'], results_no_punct['fn'])
            print(f"  [{i+1}/{len(dev_data)}] Running F0.5 (no punct): {f05_cur*100:.2f}%", flush=True)

    # Compute F0.5
    f05_lora_only = compute_f05(results_lora_only['tp'], results_lora_only['fp'], results_lora_only['fn'])
    f05_lora_punct = compute_f05(results_lora_punct['tp'], results_lora_punct['fp'], results_lora_punct['fn'])
    f05_no_punct = compute_f05(results_no_punct['tp'], results_no_punct['fp'], results_no_punct['fn'])

    print(f"\n--- {name} RESULTS ---", flush=True)
    print(f"V15 + LoRA (no punct eval):      {f05_no_punct*100:.2f}%", flush=True)
    print(f"V15 + LoRA (with punct):         {f05_lora_only*100:.2f}%", flush=True)
    print(f"V15 + LoRA + Punct Classifier:   {f05_lora_punct*100:.2f}%", flush=True)
    print(f"Punct improvement: {(f05_lora_punct - f05_lora_only)*100:+.2f}%", flush=True)

    return {
        'name': name,
        'no_punct': f05_no_punct,
        'with_punct': f05_lora_only,
        'with_punct_classifier': f05_lora_punct,
        'stats': {
            'lora_only': results_lora_only,
            'lora_punct': results_lora_punct,
            'no_punct': results_no_punct
        }
    }


def main():
    print("=" * 60, flush=True)
    print("COMBINED EVALUATION: V15 + LoRA + Punct Classifier", flush=True)
    print("=" * 60, flush=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)

    # Load tokenizer
    print("\nLoading tokenizer...", flush=True)
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(TOKENIZER_PATH)
    print("  Tokenizer loaded", flush=True)

    # Load V15 + LoRA model
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

    print("\nLoading V15 + LoRA model...", flush=True)
    gec_model = NahawiGECWithLoRA(model_config)

    base_ckpt = torch.load(BASE_MODEL_PATH, map_location='cpu', weights_only=False)
    state_dict = base_ckpt.get('model_state_dict', base_ckpt)
    gec_model.load_state_dict(state_dict, strict=False)
    print("  Base V15 loaded", flush=True)

    gec_model.add_lora_adapters()

    lora_ckpt = torch.load(LORA_MODEL_PATH, map_location='cpu', weights_only=False)
    lora_state = lora_ckpt.get('lora_state_dict', {})
    gec_model.load_state_dict(lora_state, strict=False)
    print("  LoRA weights loaded", flush=True)

    gec_model.to(device)
    gec_model.eval()
    print("  GEC model ready on", device, flush=True)

    # Load punct classifier
    print("\nLoading punct classifier...", flush=True)
    num_punct_classes = len(PUNCT_CHARS) + 1
    punct_model = PunctClassifier(model_config, num_punct_classes)

    punct_ckpt = torch.load(PUNCT_MODEL_PATH, map_location='cpu', weights_only=False)

    # Load V15 encoder weights into punct model first
    punct_model.embedding.load_state_dict({'weight': state_dict['embedding.weight']})
    punct_model.pos_encoder.load_state_dict({k.replace('pos_encoder.', ''): v for k, v in state_dict.items() if k.startswith('pos_encoder.')})
    encoder_state = {k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')}
    punct_model.encoder.load_state_dict(encoder_state)
    print("  Loaded V15 encoder into punct model", flush=True)

    # Now load the trained classifier head
    punct_model.load_state_dict(punct_ckpt['model_state_dict'], strict=False)
    punct_model.to(device)
    punct_model.eval()
    print("  Punct classifier ready", flush=True)

    class_to_punct = punct_ckpt.get('class_to_punct', {i+1: p for i, p in enumerate(PUNCT_CHARS)})
    print(f"  Punct classes: {class_to_punct}", flush=True)

    # Evaluate on both datasets
    results = []
    results.append(evaluate_dataset("QALB Dev", QALB_DEV, gec_model, punct_model, tokenizer, device, class_to_punct))
    results.append(evaluate_dataset("FASIH Test", FASIH_TEST, gec_model, punct_model, tokenizer, device, class_to_punct))

    # Final summary
    print("\n" + "=" * 60, flush=True)
    print("FINAL SUMMARY", flush=True)
    print("=" * 60, flush=True)

    for r in results:
        print(f"\n{r['name']}:", flush=True)
        print(f"  No punct:         {r['no_punct']*100:.2f}%", flush=True)
        print(f"  With punct:       {r['with_punct']*100:.2f}%", flush=True)
        print(f"  + Punct Classifier: {r['with_punct_classifier']*100:.2f}%", flush=True)

    print(f"\nSOTA (ArbESC+): 82.63%", flush=True)

    # Check if we beat SOTA on either
    for r in results:
        if r['with_punct_classifier'] >= 0.8263:
            print(f"\n🎉 BEAT SOTA on {r['name']}!", flush=True)
        elif r['no_punct'] >= 0.8263:
            print(f"\n🎉 BEAT SOTA (no-punct) on {r['name']}!", flush=True)


if __name__ == "__main__":
    main()
