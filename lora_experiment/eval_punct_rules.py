#!/usr/bin/env python3
"""
Rule-based punct insertion using QALB patterns.
Focus on high-confidence mechanical patterns only.
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

BASE_MODEL = "/home/ubuntu/nahawi/fasih_v15_model/best_model.pt"
HAMZA_LORA = "/home/ubuntu/nahawi/lora_model_hamza/epoch_1.pt"
TOKENIZER = "/home/ubuntu/nahawi/nahawi_spm.model"
QALB_DEV = "/home/ubuntu/nahawi/data/qalb_real_dev.json"

PUNCT_SET = set('،.؟!؛:,;?')

# High-confidence insertion rules from QALB patterns
# Format: (word_after, punct_to_insert, min_freq)
# Insert punct BEFORE these words
INSERT_BEFORE_RULES = [
    # Most reliable patterns (>1000 occurrences)
    ('لأن', '؛', 1650),      # semicolon before "because"
    ('لأنه', '؛', 692),     # semicolon before "because he"
    ('لأنها', '؛', 400),    # semicolon before "because she"
    ('لأنهم', '؛', 300),    # semicolon before "because they"
]

# Two-word patterns for sentence endings
# Format: (word1, word2, punct_after)
TWO_WORD_ENDINGS = [
    ('ونعم', 'الوكيل', '.'),      # 379 times
    ('شاء', 'الله', '.'),          # 354 times
    ('بإذن', 'الله', '.'),         # 274 times
    ('إلا', 'بالله', '.'),         # 199 times
    ('الرحمن', 'الرحيم', '.'),    # 168 times
    ('السلام', 'عليكم', '.'),     # 147 times
    ('رب', 'العالمين', '.'),      # 138 times
    ('إليه', 'راجعون', '.'),      # 132 times
]


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


def is_punct_token(tok):
    return all(c in PUNCT_SET for c in tok) and len(tok) > 0


def apply_punct_rules(text):
    """Apply rule-based punct insertion."""
    tokens = text.split()
    if not tokens:
        return text, {'inserted': 0, 'modified': 0}

    stats = {'inserted': 0, 'modified': 0}
    result = []

    # Build word list (non-punct only)
    words_and_idx = []
    for i, tok in enumerate(tokens):
        if not is_punct_token(tok):
            words_and_idx.append((tok, i))

    i = 0
    while i < len(tokens):
        tok = tokens[i]
        result.append(tok)

        # Check if this is a word (not punct)
        if not is_punct_token(tok):
            # Find this word's position in words_and_idx
            word_pos = None
            for wp, (w, idx) in enumerate(words_and_idx):
                if idx == i:
                    word_pos = wp
                    break

            if word_pos is not None:
                # Check two-word ending patterns
                if word_pos + 1 < len(words_and_idx):
                    next_word = words_and_idx[word_pos + 1][0]
                    for w1, w2, punct in TWO_WORD_ENDINGS:
                        if tok == w1 and next_word == w2:
                            # Check if punct already exists between current word and next word
                            # Find next word position in tokens
                            next_idx = words_and_idx[word_pos + 1][1]
                            has_punct_between = False
                            for j in range(i + 1, next_idx):
                                if is_punct_token(tokens[j]):
                                    has_punct_between = True
                                    break

                            # If we find the two-word pattern and next word has punct after
                            # Don't insert here, wait for second word
                            break

                # Check if NEXT word triggers insert-before rule
                if word_pos + 1 < len(words_and_idx):
                    next_word = words_and_idx[word_pos + 1][0]
                    # Check if there's already punct between current and next word
                    next_idx = words_and_idx[word_pos + 1][1]
                    has_punct_between = any(is_punct_token(tokens[j]) for j in range(i + 1, next_idx))

                    if not has_punct_between:
                        for trigger, punct, _ in INSERT_BEFORE_RULES:
                            if next_word == trigger:
                                result.append(punct)
                                stats['inserted'] += 1
                                break

        i += 1

    return ' '.join(result), stats


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


def main():
    print("=" * 70)
    print("EVAL: RULE-BASED PUNCT FROM QALB PATTERNS")
    print("Only inserting ؛ before لأن/لأنه/etc - most reliable pattern")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load models
    print("\nLoading models...")
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(TOKENIZER)

    config = {"vocab_size": 32000, "d_model": 768, "nhead": 12, "num_encoder_layers": 6,
              "num_decoder_layers": 6, "dim_feedforward": 3072, "dropout": 0.1, "max_seq_len": 256}

    lora_model = NahawiGEC(config)
    base_ckpt = torch.load(BASE_MODEL, map_location='cpu', weights_only=False)
    lora_model.load_state_dict(base_ckpt.get('model_state_dict', base_ckpt), strict=False)
    lora_model.add_lora(rank=64, alpha=128)
    hamza_ckpt = torch.load(HAMZA_LORA, map_location='cpu', weights_only=False)
    lora_model.load_state_dict(hamza_ckpt.get('lora_state_dict', {}), strict=False)
    lora_model.to(device).eval()
    print("Models loaded")

    with open(QALB_DEV, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"QALB dev: {len(data)} examples")

    lora_raw_wp = 0
    lora_np = 0
    rules_wp = 0
    total_stats = {'inserted': 0, 'modified': 0}

    print(f"\n{'='*60}")
    print("Evaluating...")
    print(f"{'='*60}")

    for i, item in enumerate(data):
        src = item.get('source', item.get('src', ''))
        ref = item.get('target', item.get('tgt', ''))

        # LoRA generate
        src_ids = torch.tensor([[2] + tokenizer.encode(src)[:254] + [3]], device=device)
        hyp_ids = lora_model.generate(src_ids)
        lora_hyp = tokenizer.decode(hyp_ids[0].tolist()[1:]).replace('</s>', '').strip()

        # Scores
        lora_raw_wp += compute_f05(ref, lora_hyp)
        lora_np += compute_f05(strip_punct_text(ref), strip_punct_text(lora_hyp))

        # Apply rules
        rules_hyp, stats = apply_punct_rules(lora_hyp)
        rules_wp += compute_f05(ref, rules_hyp)

        for k in total_stats:
            total_stats[k] += stats[k]

        if (i + 1) % 400 == 0:
            print(f"  [{i+1}/{len(data)}] raw_wp={100*lora_raw_wp/(i+1):.2f}% rules_wp={100*rules_wp/(i+1):.2f}%", flush=True)

    n = len(data)
    print(f"\nResults:")
    print(f"  LoRA no-punct:      {100*lora_np/n:.2f}%")
    print(f"  LoRA with-punct:    {100*lora_raw_wp/n:.2f}%")
    print(f"  Rules with-punct:   {100*rules_wp/n:.2f}%")
    print(f"  Delta:              {100*(rules_wp - lora_raw_wp)/n:+.2f}%")
    print(f"\nPunct operations: inserted={total_stats['inserted']}, modified={total_stats['modified']}")

    # Show examples where rules helped
    print("\n" + "=" * 60)
    print("SAMPLE INSERTIONS:")
    print("=" * 60)
    count = 0
    for item in data[:200]:
        src = item.get('source', item.get('src', ''))
        ref = item.get('target', item.get('tgt', ''))

        src_ids = torch.tensor([[2] + tokenizer.encode(src)[:254] + [3]], device=device)
        hyp_ids = lora_model.generate(src_ids)
        lora_hyp = tokenizer.decode(hyp_ids[0].tolist()[1:]).replace('</s>', '').strip()

        rules_hyp, stats = apply_punct_rules(lora_hyp)
        if stats['inserted'] > 0:
            print(f"\nLORA: ...{lora_hyp[max(0,len(lora_hyp)-100):]}")
            print(f"RULES: ...{rules_hyp[max(0,len(rules_hyp)-100):]}")
            print(f"REF:  ...{ref[max(0,len(ref)-100):]}")
            count += 1
            if count >= 5:
                break

    print("\n" + "=" * 60)
    print("Baseline: SOTA with-punct = 82.63%")
    print("Our LoRA with-punct = 71.34%")
    print("=" * 60)


if __name__ == "__main__":
    main()
