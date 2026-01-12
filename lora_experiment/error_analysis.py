#!/usr/bin/env python3
"""
Comprehensive Error Analysis for V15 + LoRA
Breaks down by FASIH error types + detailed QALB analysis
"""

import torch
import torch.nn as nn
import sentencepiece as spm
import json
import math
from collections import defaultdict
from dataclasses import dataclass

# Config
BASE_MODEL_PATH = "/home/ubuntu/nahawi/fasih_v15_model/best_model.pt"
LORA_MODEL_PATH = "/home/ubuntu/nahawi/lora_model_hamza/epoch_1.pt"
TOKENIZER_PATH = "/home/ubuntu/nahawi/nahawi_spm.model"
QALB_DEV = "/home/ubuntu/nahawi/data/qalb_real_dev.json"
FASIH_TEST = "/home/ubuntu/nahawi/data/fasih_test.json"

PUNCT_CHARS = set("،؛؟!.,:;?")


@dataclass
class LoRAConfig:
    base_model_path: str = ""
    tokenizer_path: str = ""
    lora_rank: int = 64
    lora_alpha: int = 128
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
    return ''.join(c for c in text if c not in PUNCT_CHARS)


def classify_error_type(src_word, tgt_word):
    """Classify error type based on FASIH categories."""
    if src_word == tgt_word:
        return None

    src_clean = strip_punct(src_word)
    tgt_clean = strip_punct(tgt_word)

    if src_clean == tgt_clean:
        return 'punct_only'

    # Hamza variations
    hamza_chars = set('أإآءؤئا')
    src_hamza = set(src_clean) & hamza_chars
    tgt_hamza = set(tgt_clean) & hamza_chars
    if src_hamza != tgt_hamza:
        # Specific hamza subcategories
        if 'ا' in src_clean and 'أ' in tgt_clean:
            return 'hamza_add_alif'
        if 'ا' in src_clean and 'إ' in tgt_clean:
            return 'hamza_add_alif_kasra'
        if 'أ' in src_clean and 'إ' in tgt_clean:
            return 'hamza_alif_swap'
        if 'ؤ' in src_clean or 'ؤ' in tgt_clean:
            return 'hamza_waw'
        if 'ئ' in src_clean or 'ئ' in tgt_clean:
            return 'hamza_yaa'
        if 'ء' in src_clean or 'ء' in tgt_clean:
            return 'hamza_standalone'
        if 'آ' in src_clean or 'آ' in tgt_clean:
            return 'alif_madda'
        return 'hamza_other'

    # Taa marbuta
    if ('ة' in src_clean) != ('ة' in tgt_clean) or ('ه' in src_clean and 'ة' in tgt_clean):
        return 'taa_marbuta'

    # Alif maqsura
    if ('ى' in src_clean) != ('ى' in tgt_clean) or ('ي' in src_clean and 'ى' in tgt_clean) or ('ى' in src_clean and 'ي' in tgt_clean):
        return 'alif_maqsura'

    # Letter confusions
    if ('ض' in src_clean and 'ظ' in tgt_clean) or ('ظ' in src_clean and 'ض' in tgt_clean):
        return 'dad_za_confusion'
    if ('د' in src_clean and 'ذ' in tgt_clean) or ('ذ' in src_clean and 'د' in tgt_clean):
        return 'dal_thal_confusion'
    if ('ت' in src_clean and 'ط' in tgt_clean) or ('ط' in src_clean and 'ت' in tgt_clean):
        return 'taa_ta_confusion'
    if ('س' in src_clean and 'ص' in tgt_clean) or ('ص' in src_clean and 'س' in tgt_clean):
        return 'seen_sad_confusion'

    # Length-based categories
    len_diff = len(tgt_clean) - len(src_clean)

    if len_diff == 0 and len(src_clean) > 0:
        # Same length - substitution
        diff_count = sum(1 for a, b in zip(src_clean, tgt_clean) if a != b)
        if diff_count == 1:
            return 'single_char_subst'
        return 'multi_char_subst'

    if len_diff == 1:
        # One char insertion needed
        return 'one_char_ins'

    if len_diff == -1:
        # One char deletion needed
        return 'one_char_del'

    if len_diff > 1:
        # Check for missing doubled letter
        for i, c in enumerate(tgt_clean):
            if i < len(tgt_clean) - 1 and tgt_clean[i] == tgt_clean[i+1]:
                test = tgt_clean[:i] + tgt_clean[i+1:]
                if test == src_clean:
                    return 'missing_double_letter'
        return 'multi_char_ins'

    if len_diff < -1:
        return 'multi_char_del'

    # Space/split issues
    if ' ' in src_clean or ' ' in tgt_clean:
        return 'spacing'

    return 'other_edit'


def analyze_errors(src, ref, hyp):
    """Analyze errors between source, reference, and hypothesis."""
    src_words = strip_punct(src).split()
    ref_words = strip_punct(ref).split()
    hyp_words = strip_punct(hyp).split()

    errors = {
        'by_type': defaultdict(lambda: {'total': 0, 'correct': 0, 'missed': 0, 'wrong': 0}),
        'examples': []
    }

    # Align and compare
    min_len = min(len(src_words), len(ref_words), len(hyp_words))

    for i in range(min_len):
        src_w = src_words[i]
        ref_w = ref_words[i]
        hyp_w = hyp_words[i]

        error_type = classify_error_type(src_w, ref_w)

        if error_type:
            errors['by_type'][error_type]['total'] += 1

            if hyp_w == ref_w:
                errors['by_type'][error_type]['correct'] += 1
            elif hyp_w == src_w:
                errors['by_type'][error_type]['missed'] += 1
                if len(errors['examples']) < 50:
                    errors['examples'].append({
                        'type': error_type,
                        'src': src_w,
                        'ref': ref_w,
                        'hyp': hyp_w,
                        'status': 'MISSED'
                    })
            else:
                errors['by_type'][error_type]['wrong'] += 1
                if len(errors['examples']) < 50:
                    errors['examples'].append({
                        'type': error_type,
                        'src': src_w,
                        'ref': ref_w,
                        'hyp': hyp_w,
                        'status': 'WRONG'
                    })

    return errors


def main():
    print("=" * 70, flush=True)
    print("V15 + LoRA ERROR ANALYSIS", flush=True)
    print("=" * 70, flush=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)

    # Load model
    print("\nLoading model...", flush=True)
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(TOKENIZER_PATH)

    model_config = {
        "vocab_size": 32000, "d_model": 768, "nhead": 12,
        "num_encoder_layers": 6, "num_decoder_layers": 6,
        "dim_feedforward": 3072, "dropout": 0.1, "max_seq_len": 256,
    }

    model = NahawiGECWithLoRA(model_config)
    base_ckpt = torch.load(BASE_MODEL_PATH, map_location='cpu', weights_only=False)
    state_dict = base_ckpt.get('model_state_dict', base_ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.add_lora_adapters()
    lora_ckpt = torch.load(LORA_MODEL_PATH, map_location='cpu', weights_only=False)
    model.load_state_dict(lora_ckpt.get('lora_state_dict', {}), strict=False)
    model.to(device).eval()
    print("  Model loaded", flush=True)

    # Analyze both datasets
    for dataset_name, dataset_path in [("QALB Dev", QALB_DEV), ("FASIH Test", FASIH_TEST)]:
        print(f"\n{'='*70}", flush=True)
        print(f"ANALYZING: {dataset_name}", flush=True)
        print(f"{'='*70}", flush=True)

        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.loads(f.read().strip())
        print(f"Loaded {len(data)} examples", flush=True)

        all_errors = defaultdict(lambda: {'total': 0, 'correct': 0, 'missed': 0, 'wrong': 0})
        all_examples = []

        for i, item in enumerate(data):
            src = item.get('source', item.get('src', ''))
            ref = item.get('target', item.get('tgt', ''))

            if not src or not ref:
                continue

            # Generate
            src_ids = [2] + tokenizer.encode(src)[:254] + [3]
            src_tensor = torch.tensor([src_ids], device=device)
            hyp_ids = model.generate(src_tensor, max_len=256)
            hyp = tokenizer.decode(hyp_ids[0].tolist()[1:]).replace('</s>', '').strip()

            # Analyze
            errors = analyze_errors(src, ref, hyp)

            for err_type, counts in errors['by_type'].items():
                all_errors[err_type]['total'] += counts['total']
                all_errors[err_type]['correct'] += counts['correct']
                all_errors[err_type]['missed'] += counts['missed']
                all_errors[err_type]['wrong'] += counts['wrong']

            all_examples.extend(errors['examples'])

            if (i + 1) % 500 == 0:
                print(f"  Processed {i+1}/{len(data)}", flush=True)

        # Print results
        print(f"\n{'='*70}", flush=True)
        print(f"{dataset_name} ERROR BREAKDOWN", flush=True)
        print(f"{'='*70}", flush=True)
        print(f"\n{'Error Type':<25} {'Total':>8} {'Correct':>8} {'Missed':>8} {'Wrong':>8} {'Acc%':>8}", flush=True)
        print("-" * 70, flush=True)

        sorted_errors = sorted(all_errors.items(), key=lambda x: x[1]['total'], reverse=True)

        for err_type, counts in sorted_errors:
            total = counts['total']
            correct = counts['correct']
            missed = counts['missed']
            wrong = counts['wrong']
            acc = (correct / total * 100) if total > 0 else 0
            print(f"{err_type:<25} {total:>8} {correct:>8} {missed:>8} {wrong:>8} {acc:>7.1f}%", flush=True)

        # Summary
        total_all = sum(c['total'] for c in all_errors.values())
        correct_all = sum(c['correct'] for c in all_errors.values())
        print("-" * 70, flush=True)
        print(f"{'TOTAL':<25} {total_all:>8} {correct_all:>8}", flush=True)
        print(f"Overall Accuracy: {correct_all/total_all*100:.1f}%", flush=True)

        # Worst performers (for targeting)
        print(f"\n--- WORST PERFORMERS (potential 9 points) ---", flush=True)
        for err_type, counts in sorted_errors:
            total = counts['total']
            correct = counts['correct']
            missed = counts['missed']
            acc = (correct / total * 100) if total > 0 else 0
            if acc < 70 and total >= 20:
                potential = (total - correct) / total_all * 100
                print(f"  {err_type}: {acc:.1f}% acc, {missed} missed ({potential:.1f}% of total errors)", flush=True)

        # Example errors
        print(f"\n--- SAMPLE MISSED/WRONG CORRECTIONS ---", flush=True)
        for ex in all_examples[:20]:
            print(f"  [{ex['type']}] {ex['status']}: '{ex['src']}' -> gold:'{ex['ref']}' hyp:'{ex['hyp']}'", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("ANALYSIS COMPLETE", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
