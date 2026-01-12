#!/usr/bin/env python3
"""
Diagnose what the punct classifier is doing wrong.
Look at actual examples to understand the failure mode.
"""

import torch
import torch.nn as nn
import sentencepiece as spm
import json
import math
from dataclasses import dataclass

# Config
BASE_MODEL_PATH = "/home/ubuntu/nahawi/fasih_v15_model/best_model.pt"
LORA_MODEL_PATH = "/home/ubuntu/nahawi/lora_model_v15/best_model.pt"
PUNCT_MODEL_PATH = "/home/ubuntu/nahawi/punct_classifier_v15/best_model.pt"
TOKENIZER_PATH = "/home/ubuntu/nahawi/nahawi_spm.model"
QALB_DEV = "/home/ubuntu/nahawi/data/qalb_real_dev.json"

PUNCT_CHARS = "،؛؟!.,:;?"
PUNCT_SET = set(PUNCT_CHARS)


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
            d_model=config["d_model"], nhead=config["nhead"],
            dim_feedforward=config["dim_feedforward"], dropout=config["dropout"], batch_first=True)
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


def count_punct(text):
    return sum(1 for c in text if c in PUNCT_SET)


def get_punct_positions(text):
    """Return list of (position, punct_char) for all punct in text."""
    positions = []
    for i, c in enumerate(text):
        if c in PUNCT_SET:
            positions.append((i, c))
    return positions


def apply_punct_classifier_with_details(text, punct_model, tokenizer, device, class_to_punct):
    """Apply punct classifier and return details about what it did."""
    tokens = tokenizer.encode(text)[:254]
    src_ids = torch.tensor([[2] + tokens + [3]], device=device)

    with torch.no_grad():
        logits = punct_model(src_ids)
        probs = torch.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)[0].cpu().tolist()
        confidences = probs.max(dim=-1)[0][0].cpu().tolist()

    pieces = [tokenizer.id_to_piece(t) for t in tokens]
    result_pieces = []
    punct_decisions = []

    for i, piece in enumerate(pieces):
        clean_piece = piece
        for p in PUNCT_CHARS:
            clean_piece = clean_piece.replace(p, '')
        result_pieces.append(clean_piece)

        pred_class = preds[i + 1]  # +1 for BOS
        conf = confidences[i + 1]

        if pred_class > 0 and pred_class in class_to_punct:
            punct_char = class_to_punct[pred_class]
            result_pieces.append(punct_char)
            punct_decisions.append({
                'position': i,
                'piece': piece,
                'punct': punct_char,
                'confidence': conf,
                'action': 'ADD'
            })
        elif pred_class == 0:
            punct_decisions.append({
                'position': i,
                'piece': piece,
                'punct': None,
                'confidence': conf,
                'action': 'NONE'
            })

    return tokenizer.decode_pieces(result_pieces), punct_decisions


def main():
    print("=" * 70, flush=True)
    print("PUNCT CLASSIFIER DIAGNOSIS", flush=True)
    print("=" * 70, flush=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)

    # Load everything
    print("\nLoading models...", flush=True)
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(TOKENIZER_PATH)

    model_config = {
        "vocab_size": 32000, "d_model": 768, "nhead": 12,
        "num_encoder_layers": 6, "num_decoder_layers": 6,
        "dim_feedforward": 3072, "dropout": 0.1, "max_seq_len": 256,
    }

    # Load GEC model
    gec_model = NahawiGECWithLoRA(model_config)
    base_ckpt = torch.load(BASE_MODEL_PATH, map_location='cpu', weights_only=False)
    state_dict = base_ckpt.get('model_state_dict', base_ckpt)
    gec_model.load_state_dict(state_dict, strict=False)
    gec_model.add_lora_adapters()
    lora_ckpt = torch.load(LORA_MODEL_PATH, map_location='cpu', weights_only=False)
    gec_model.load_state_dict(lora_ckpt.get('lora_state_dict', {}), strict=False)
    gec_model.to(device).eval()

    # Load punct classifier
    num_punct_classes = len(PUNCT_CHARS) + 1
    punct_model = PunctClassifier(model_config, num_punct_classes)
    punct_ckpt = torch.load(PUNCT_MODEL_PATH, map_location='cpu', weights_only=False)
    punct_model.embedding.load_state_dict({'weight': state_dict['embedding.weight']})
    punct_model.pos_encoder.load_state_dict({k.replace('pos_encoder.', ''): v for k, v in state_dict.items() if k.startswith('pos_encoder.')})
    encoder_state = {k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')}
    punct_model.encoder.load_state_dict(encoder_state)
    punct_model.load_state_dict(punct_ckpt['model_state_dict'], strict=False)
    punct_model.to(device).eval()

    class_to_punct = punct_ckpt.get('class_to_punct', {i+1: p for i, p in enumerate(PUNCT_CHARS)})
    print(f"Punct mapping: {class_to_punct}", flush=True)

    # Load data
    with open(QALB_DEV, 'r', encoding='utf-8') as f:
        dev_data = json.loads(f.read().strip())

    print(f"\nAnalyzing first 20 examples...\n", flush=True)

    # Stats
    total_lora_punct = 0
    total_classifier_punct = 0
    total_gold_punct = 0
    total_src_punct = 0

    for i, item in enumerate(dev_data[:20]):
        src = item.get('source', item.get('src', ''))
        ref = item.get('target', item.get('tgt', ''))

        # Generate with LoRA
        src_ids = [2] + tokenizer.encode(src)[:254] + [3]
        src_tensor = torch.tensor([src_ids], device=device)
        hyp_ids = gec_model.generate(src_tensor, max_len=256)
        hyp_lora = tokenizer.decode(hyp_ids[0].tolist()[1:]).replace('</s>', '').strip()

        # Apply classifier
        hyp_classifier, decisions = apply_punct_classifier_with_details(
            hyp_lora, punct_model, tokenizer, device, class_to_punct)

        # Count punct
        src_punct = count_punct(src)
        gold_punct = count_punct(ref)
        lora_punct = count_punct(hyp_lora)
        classifier_punct = count_punct(hyp_classifier)

        total_src_punct += src_punct
        total_gold_punct += gold_punct
        total_lora_punct += lora_punct
        total_classifier_punct += classifier_punct

        # Show examples where classifier made things worse
        punct_added = [d for d in decisions if d['action'] == 'ADD']

        print(f"{'='*70}", flush=True)
        print(f"Example {i+1}:", flush=True)
        print(f"  SRC punct: {src_punct} | GOLD punct: {gold_punct} | LoRA punct: {lora_punct} | Classifier punct: {classifier_punct}", flush=True)
        print(f"  SRC:        {src[:100]}...", flush=True)
        print(f"  GOLD:       {ref[:100]}...", flush=True)
        print(f"  LoRA:       {hyp_lora[:100]}...", flush=True)
        print(f"  Classifier: {hyp_classifier[:100]}...", flush=True)

        if punct_added:
            print(f"  Punct added by classifier ({len(punct_added)} total):", flush=True)
            for d in punct_added[:5]:
                print(f"    '{d['punct']}' after '{d['piece']}' (conf: {d['confidence']:.3f})", flush=True)

    print(f"\n{'='*70}", flush=True)
    print("AGGREGATE STATS (first 20 examples):", flush=True)
    print(f"  Source punct total:     {total_src_punct}", flush=True)
    print(f"  Gold punct total:       {total_gold_punct}", flush=True)
    print(f"  LoRA punct total:       {total_lora_punct}", flush=True)
    print(f"  Classifier punct total: {total_classifier_punct}", flush=True)
    print(f"", flush=True)
    print(f"  LoRA vs Gold:       {total_lora_punct - total_gold_punct:+d} punct", flush=True)
    print(f"  Classifier vs Gold: {total_classifier_punct - total_gold_punct:+d} punct", flush=True)
    print(f"  Classifier vs LoRA: {total_classifier_punct - total_lora_punct:+d} punct", flush=True)


if __name__ == "__main__":
    main()
