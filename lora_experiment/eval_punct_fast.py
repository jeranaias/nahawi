#!/usr/bin/env python3
"""
Fast eval: LoRA + Punct Classifier pipeline.
Batched punct prediction for speed.
"""

import torch
import torch.nn as nn
import sentencepiece as spm
import json
import math
import pickle
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
PUNCT_MODEL = "/home/ubuntu/nahawi/punct_classifier_qalb/best_model.pt"
PUNCT_VOCAB = "/home/ubuntu/nahawi/punct_classifier_qalb/vocab.pkl"

PUNCT_SET = set('،.؟!؛:,;?')
PUNCT_CHARS = ['،', '.', '؟', '!', '؛', ':']
NUM_PUNCT = 6
NONE_IDX = 6
CONTEXT_SIZE = 2

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


class PunctClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, context_size=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        input_dim = embed_dim * (context_size + 1)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, NUM_PUNCT + 1)

    def forward(self, context):
        embedded = self.embedding(context)
        flat = embedded.view(embedded.size(0), -1)
        h = torch.relu(self.fc1(flat))
        h = self.dropout(h)
        h = torch.relu(self.fc2(h))
        return self.fc3(h)


def strip_punct_words(text):
    """Remove punct tokens from text."""
    return [w for w in text.split() if not all(c in PUNCT_SET for c in w)]


def apply_punct_batched(words, punct_model, vocab, device, threshold=0.5):
    """Apply punct classifier in batch to all words."""
    if not words:
        return ""

    # Build all contexts at once
    contexts = []
    for i in range(len(words)):
        start = max(0, i - CONTEXT_SIZE)
        ctx = words[start:i+1]
        while len(ctx) < CONTEXT_SIZE + 1:
            ctx.insert(0, '<PAD>')
        contexts.append([vocab.get(w, 1) for w in ctx])

    # Batch predict
    context_tensor = torch.tensor(contexts, device=device)
    with torch.no_grad():
        logits = punct_model(context_tensor)
        probs = torch.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1).cpu().tolist()
        confs = probs.max(dim=-1).values.cpu().tolist()

    # Build result
    result = []
    for i, word in enumerate(words):
        result.append(word)
        if preds[i] != NONE_IDX and confs[i] > threshold:
            result.append(PUNCT_CHARS[preds[i]])

    return ' '.join(result)


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
    print("FAST EVAL: LoRA + PUNCT CLASSIFIER")
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

    with open(PUNCT_VOCAB, 'rb') as f:
        punct_vocab = pickle.load(f)
    punct_model = PunctClassifier(len(punct_vocab)).to(device)
    punct_model.load_state_dict(torch.load(PUNCT_MODEL, map_location=device))
    punct_model.eval()
    print("Models loaded")

    with open(QALB_DEV, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"QALB dev: {len(data)} examples")

    # Test thresholds
    for threshold in [0.3, 0.5, 0.7]:
        print(f"\n{'='*60}")
        print(f"Threshold: {threshold}")
        print(f"{'='*60}")

        lora_raw_wp = 0
        lora_np = 0
        pipeline_wp = 0

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

            # Pipeline: strip punct then re-add with classifier
            words = strip_punct_words(lora_hyp)
            pipeline_hyp = apply_punct_batched(words, punct_model, punct_vocab, device, threshold)
            pipeline_wp += compute_f05(ref, pipeline_hyp)

            if (i + 1) % 400 == 0:
                print(f"  [{i+1}/{len(data)}] raw_wp={100*lora_raw_wp/(i+1):.2f}% pipe_wp={100*pipeline_wp/(i+1):.2f}%", flush=True)

        n = len(data)
        print(f"\nResults:")
        print(f"  LoRA no-punct:      {100*lora_np/n:.2f}%")
        print(f"  LoRA with-punct:    {100*lora_raw_wp/n:.2f}%")
        print(f"  Pipeline with-punct: {100*pipeline_wp/n:.2f}%")
        print(f"  Delta:              {100*(pipeline_wp - lora_raw_wp)/n:+.2f}%")

    print("\n" + "=" * 60)
    print("Baseline: SOTA with-punct = 82.63%")
    print("Our LoRA with-punct = 71.34%")
    print("=" * 60)


if __name__ == "__main__":
    main()
