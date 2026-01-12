#!/usr/bin/env python3
"""
Deep punctuation analysis on QALB dev.
Understand why we're -11.3 points behind SOTA on with-punct.
"""

import torch
import torch.nn as nn
import sentencepiece as spm
import json
import math
from collections import defaultdict, Counter
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

PUNCT_CHARS = ['،', '.', '؟', '!', '؛', ':', ',', ';', '?']
PUNCT_SET = set(PUNCT_CHARS)

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


def count_punct(text):
    """Count punctuation by type."""
    counts = Counter()
    for c in text:
        if c in PUNCT_SET:
            counts[c] += 1
    return counts


def get_punct_positions(text):
    """Get positions of punctuation in text."""
    positions = []
    for i, c in enumerate(text):
        if c in PUNCT_SET:
            positions.append((i, c))
    return positions


def analyze_punct_errors(src, gold, hyp):
    """Analyze punctuation errors between gold and hypothesis."""
    gold_punct = get_punct_positions(gold)
    hyp_punct = get_punct_positions(hyp)

    errors = {
        'correct': [],      # Right punct at right position
        'wrong_type': [],   # Punct at right position but wrong type
        'inserted': [],     # Punct in hyp but not in gold (false positive)
        'missed': [],       # Punct in gold but not in hyp (false negative)
        'wrong_pos': [],    # Right punct but wrong position
    }

    gold_pos_set = {p[0] for p in gold_punct}
    hyp_pos_set = {p[0] for p in hyp_punct}

    # Check each gold punct
    for pos, pchar in gold_punct:
        # Find if hyp has punct nearby (within 3 chars)
        found = False
        for hpos, hchar in hyp_punct:
            if abs(hpos - pos) <= 3:
                if hpos == pos and hchar == pchar:
                    errors['correct'].append((pos, pchar))
                elif hpos == pos and hchar != pchar:
                    errors['wrong_type'].append((pos, pchar, hchar))
                elif hchar == pchar:
                    errors['wrong_pos'].append((pos, hpos, pchar))
                found = True
                break
        if not found:
            errors['missed'].append((pos, pchar))

    # Check for insertions (in hyp but not near any gold)
    for hpos, hchar in hyp_punct:
        is_insertion = True
        for gpos, gchar in gold_punct:
            if abs(hpos - gpos) <= 3:
                is_insertion = False
                break
        if is_insertion:
            errors['inserted'].append((hpos, hchar))

    return errors


def main():
    print("=" * 70)
    print("PUNCTUATION ANALYSIS ON QALB DEV")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(TOKENIZER)

    config = {"vocab_size": 32000, "d_model": 768, "nhead": 12, "num_encoder_layers": 6,
              "num_decoder_layers": 6, "dim_feedforward": 3072, "dropout": 0.1, "max_seq_len": 256}

    print("\nLoading model (Hamza LoRA epoch 1)...")
    model = NahawiGEC(config)
    base_ckpt = torch.load(BASE_MODEL, map_location='cpu', weights_only=False)
    model.load_state_dict(base_ckpt.get('model_state_dict', base_ckpt), strict=False)
    model.add_lora(rank=64, alpha=128)
    hamza_ckpt = torch.load(HAMZA_LORA, map_location='cpu', weights_only=False)
    model.load_state_dict(hamza_ckpt.get('lora_state_dict', {}), strict=False)
    model.to(device).eval()
    print("Model ready")

    with open(QALB_DEV, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} examples")

    # Aggregate stats
    src_punct_total = Counter()
    gold_punct_total = Counter()
    hyp_punct_total = Counter()

    all_errors = {
        'correct': 0,
        'wrong_type': 0,
        'inserted': 0,
        'missed': 0,
        'wrong_pos': 0,
    }

    error_samples = {
        'inserted': [],
        'missed': [],
        'wrong_type': [],
        'wrong_pos': [],
    }

    print("\nProcessing...")
    for i, item in enumerate(data):
        src = item.get('source', item.get('src', ''))
        gold = item.get('target', item.get('tgt', ''))

        src_ids = torch.tensor([[2] + tokenizer.encode(src)[:254] + [3]], device=device)
        hyp_ids = model.generate(src_ids)
        hyp = tokenizer.decode(hyp_ids[0].tolist()[1:]).replace('</s>', '').strip()

        # Count punct
        src_punct_total.update(count_punct(src))
        gold_punct_total.update(count_punct(gold))
        hyp_punct_total.update(count_punct(hyp))

        # Analyze errors
        errors = analyze_punct_errors(src, gold, hyp)
        all_errors['correct'] += len(errors['correct'])
        all_errors['wrong_type'] += len(errors['wrong_type'])
        all_errors['inserted'] += len(errors['inserted'])
        all_errors['missed'] += len(errors['missed'])
        all_errors['wrong_pos'] += len(errors['wrong_pos'])

        # Collect samples
        if errors['inserted'] and len(error_samples['inserted']) < 10:
            for pos, pchar in errors['inserted'][:2]:
                context_start = max(0, pos - 20)
                context_end = min(len(hyp), pos + 20)
                error_samples['inserted'].append({
                    'hyp_context': hyp[context_start:context_end],
                    'gold_context': gold[context_start:context_end] if context_end <= len(gold) else gold[-40:],
                    'punct': pchar
                })

        if errors['missed'] and len(error_samples['missed']) < 10:
            for pos, pchar in errors['missed'][:2]:
                context_start = max(0, pos - 20)
                context_end = min(len(gold), pos + 20)
                error_samples['missed'].append({
                    'gold_context': gold[context_start:context_end],
                    'hyp_context': hyp[context_start:context_end] if context_end <= len(hyp) else hyp[-40:],
                    'punct': pchar
                })

        if (i + 1) % 500 == 0:
            print(f"  [{i+1}/{len(data)}]", flush=True)

    # Print results
    print("\n" + "=" * 70)
    print("PUNCTUATION COUNTS")
    print("=" * 70)

    print(f"\n{'Punct':<6} {'Source':>10} {'Gold':>10} {'Hyp':>10} {'Hyp-Gold':>10}")
    print("-" * 50)

    all_punct = set(src_punct_total.keys()) | set(gold_punct_total.keys()) | set(hyp_punct_total.keys())
    for p in sorted(all_punct, key=lambda x: gold_punct_total.get(x, 0), reverse=True):
        s = src_punct_total.get(p, 0)
        g = gold_punct_total.get(p, 0)
        h = hyp_punct_total.get(p, 0)
        diff = h - g
        diff_str = f"+{diff}" if diff > 0 else str(diff)
        print(f"'{p}'    {s:>10} {g:>10} {h:>10} {diff_str:>10}")

    src_total = sum(src_punct_total.values())
    gold_total = sum(gold_punct_total.values())
    hyp_total = sum(hyp_punct_total.values())
    print("-" * 50)
    print(f"TOTAL  {src_total:>10} {gold_total:>10} {hyp_total:>10} {hyp_total - gold_total:>+10}")

    print("\n" + "=" * 70)
    print("ERROR ANALYSIS")
    print("=" * 70)

    total_gold_punct = all_errors['correct'] + all_errors['missed'] + all_errors['wrong_type'] + all_errors['wrong_pos']

    print(f"\nCorrect punct:     {all_errors['correct']:>6} ({100*all_errors['correct']/total_gold_punct:.1f}% of gold)")
    print(f"Wrong type:        {all_errors['wrong_type']:>6} ({100*all_errors['wrong_type']/total_gold_punct:.1f}% of gold)")
    print(f"Wrong position:    {all_errors['wrong_pos']:>6} ({100*all_errors['wrong_pos']/total_gold_punct:.1f}% of gold)")
    print(f"Missed (FN):       {all_errors['missed']:>6} ({100*all_errors['missed']/total_gold_punct:.1f}% of gold)")
    print(f"Inserted (FP):     {all_errors['inserted']:>6} (spurious)")

    print(f"\nPrecision: {100*all_errors['correct']/(all_errors['correct']+all_errors['inserted']):.1f}%")
    print(f"Recall:    {100*all_errors['correct']/total_gold_punct:.1f}%")

    print("\n" + "=" * 70)
    print("SAMPLE ERRORS")
    print("=" * 70)

    print("\n--- INSERTED (False Positives) ---")
    for i, sample in enumerate(error_samples['inserted'][:5]):
        print(f"\n{i+1}. Inserted '{sample['punct']}'")
        print(f"   Hyp:  ...{sample['hyp_context']}...")
        print(f"   Gold: ...{sample['gold_context']}...")

    print("\n--- MISSED (False Negatives) ---")
    for i, sample in enumerate(error_samples['missed'][:5]):
        print(f"\n{i+1}. Missed '{sample['punct']}'")
        print(f"   Gold: ...{sample['gold_context']}...")
        print(f"   Hyp:  ...{sample['hyp_context']}...")

    # Over/under insertion summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if hyp_total > gold_total:
        print(f"\nOVER-INSERTION: Model produces {hyp_total - gold_total} more punct than gold (+{100*(hyp_total-gold_total)/gold_total:.1f}%)")
    else:
        print(f"\nUNDER-INSERTION: Model produces {gold_total - hyp_total} fewer punct than gold ({100*(hyp_total-gold_total)/gold_total:.1f}%)")

    print(f"\nKey issues:")
    if all_errors['inserted'] > all_errors['missed']:
        print(f"  - More insertions ({all_errors['inserted']}) than misses ({all_errors['missed']}) → PRECISION problem")
    else:
        print(f"  - More misses ({all_errors['missed']}) than insertions ({all_errors['inserted']}) → RECALL problem")


if __name__ == "__main__":
    main()
