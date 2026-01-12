#!/usr/bin/env python3
"""
Competitive Comparison Test Runner.
Runs Nahawi on the competitive test set and outputs results for comparison with Word/Google.
"""

import torch
import torch.nn as nn
import sentencepiece as spm
import math
import json
from pathlib import Path
from datetime import datetime

# Model configuration
CONFIG = {
    'vocab_size': 32000, 'd_model': 768, 'nhead': 12,
    'num_encoder_layers': 6, 'num_decoder_layers': 6,
    'dim_feedforward': 3072, 'dropout': 0.1, 'max_seq_len': 256
}


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


class NahawiGEC(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config['vocab_size'], config['d_model'])
        self.pos_encoder = PositionalEncoding(config['d_model'], config['max_seq_len'])
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config['d_model'], nhead=config['nhead'],
            dim_feedforward=config['dim_feedforward'],
            dropout=config['dropout'], batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config['num_encoder_layers'])
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config['d_model'], nhead=config['nhead'],
            dim_feedforward=config['dim_feedforward'],
            dropout=config['dropout'], batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config['num_decoder_layers'])
        self.output_projection = nn.Linear(config['d_model'], config['vocab_size'])
        self.output_projection.weight = self.embedding.weight
        self.d_model = config['d_model']

    @torch.no_grad()
    def generate(self, src_ids, max_len=256, eos_id=3):
        self.eval()
        device = src_ids.device
        src_emb = self.pos_encoder(self.embedding(src_ids) * math.sqrt(self.d_model))
        memory = self.encoder(src_emb)
        generated = torch.full((1, 1), 2, dtype=torch.long, device=device)
        for _ in range(max_len - 1):
            tgt_emb = self.pos_encoder(self.embedding(generated) * math.sqrt(self.d_model))
            causal_mask = nn.Transformer.generate_square_subsequent_mask(generated.size(1), device=device)
            output = self.decoder(tgt_emb, memory, tgt_mask=causal_mask)
            logits = self.output_projection(output[:, -1, :])
            next_token = logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if next_token.item() == eos_id:
                break
        return generated


class RulePostProcessor:
    def __init__(self):
        self.maqsura_fix = {'الي': 'إلى', 'علي': 'على', 'حتي': 'حتى', 'متي': 'متى'}
        self.taa_fixes = {'مدرسه': 'مدرسة', 'جامعه': 'جامعة', 'مدينه': 'مدينة'}

    def process(self, text):
        words = text.split()
        result = []
        for word in words:
            clean = word.rstrip('،.؟!؛:')
            punct = word[len(clean):]
            if clean in self.maqsura_fix:
                result.append(self.maqsura_fix[clean] + punct)
            elif clean in self.taa_fixes:
                result.append(self.taa_fixes[clean] + punct)
            else:
                result.append(word)
        return ' '.join(result)


def load_model(model_path, spm_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sp = spm.SentencePieceProcessor()
    sp.Load(spm_path)
    model = NahawiGEC(CONFIG).to(device)
    state = torch.load(model_path, map_location=device)
    if 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'])
    else:
        model.load_state_dict(state)
    model.eval()
    rules = RulePostProcessor()
    return model, sp, rules, device


def correct(text, model, sp, rules, device):
    src_ids = torch.tensor([[2] + sp.EncodeAsIds(text)[:254] + [3]], device=device)
    output = model.generate(src_ids)
    ids = output[0].tolist()
    if 3 in ids:
        ids = ids[:ids.index(3)]
    if ids and ids[0] == 2:
        ids = ids[1:]
    decoded = sp.DecodeIds(ids)
    return rules.process(decoded)


def normalize(text):
    """Normalize for comparison - remove trailing punctuation added by model."""
    return text.rstrip(' .،؟!')


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-file', default='benchmark/competitive_test_set.json')
    parser.add_argument('--model', default='models/v5_hybrid/fasih_v5_best.pt')
    parser.add_argument('--spm', default='models/v5_hybrid/nahawi_spm.model')
    parser.add_argument('--output', default='benchmark/nahawi_competitive_results.json')
    args = parser.parse_args()

    print("=" * 70)
    print("NAHAWI COMPETITIVE COMPARISON TEST")
    print("=" * 70)

    # Load test set
    with open(args.test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    print(f"\nLoaded {len(test_data['test_cases'])} test cases")

    # Load model
    print("Loading model...")
    model, sp, rules, device = load_model(args.model, args.spm)
    print(f"Model loaded on {device}")

    # Run corrections
    results = []
    correct_count = 0
    partial_count = 0
    error_stats = {}

    print("\nRunning corrections...")
    for tc in test_data['test_cases']:
        source = tc['source']
        target = tc['target']
        hypothesis = correct(source, model, sp, rules, device)

        # Normalize for comparison
        hyp_norm = normalize(hypothesis)
        tgt_norm = normalize(target)

        # Check if correct
        is_correct = hyp_norm == tgt_norm

        # Check for partial match (some errors fixed)
        src_words = source.split()
        tgt_words = target.split()
        hyp_words = hyp_norm.split()

        errors_in_src = sum(1 for s, t in zip(src_words, tgt_words) if s != t)
        errors_fixed = sum(1 for s, t, h in zip(src_words, tgt_words, hyp_words) if s != t and h == t)

        if is_correct:
            correct_count += 1
            status = "CORRECT"
        elif errors_fixed > 0:
            partial_count += 1
            status = f"PARTIAL ({errors_fixed}/{errors_in_src})"
        else:
            status = "MISSED"

        # Track by error type
        for err_type in tc.get('errors', []):
            if err_type not in error_stats:
                error_stats[err_type] = {'total': 0, 'fixed': 0}
            error_stats[err_type]['total'] += 1

        result = {
            'id': tc['id'],
            'source': source,
            'target': target,
            'hypothesis': hypothesis,
            'hypothesis_normalized': hyp_norm,
            'is_correct': is_correct,
            'errors_in_source': errors_in_src,
            'errors_fixed': errors_fixed,
            'error_types': tc.get('errors', []),
            'status': status
        }
        results.append(result)

        if tc['id'] <= 10 or not is_correct:
            print(f"\n[{tc['id']}] {status}")
            if not is_correct:
                print(f"  SRC: {source}")
                print(f"  TGT: {target}")
                print(f"  HYP: {hyp_norm}")

    # Calculate statistics
    total = len(results)
    sentences_with_errors = sum(1 for r in results if r['errors_in_source'] > 0)

    accuracy = correct_count / total * 100
    error_correction_rate = (correct_count + partial_count) / sentences_with_errors * 100 if sentences_with_errors > 0 else 0

    print("\n" + "=" * 70)
    print("NAHAWI RESULTS")
    print("=" * 70)
    print(f"\nExact Match Accuracy: {correct_count}/{total} ({accuracy:.1f}%)")
    print(f"Partial/Full Correction: {correct_count + partial_count}/{sentences_with_errors} ({error_correction_rate:.1f}%)")
    print(f"\nClean sentences preserved: {sum(1 for r in results if r['errors_in_source'] == 0 and r['is_correct'])}")

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'model': args.model,
        'summary': {
            'total_sentences': total,
            'exact_match': correct_count,
            'partial_match': partial_count,
            'accuracy_percent': accuracy,
            'error_correction_rate': error_correction_rate
        },
        'results': results
    }

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {args.output}")

    # Generate Word/Google test file
    word_test_file = args.output.replace('.json', '_for_word_google.txt')
    with open(word_test_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("COMPETITIVE TEST: Copy these sentences into Word/Google Docs\n")
        f.write("Record which errors each tool catches\n")
        f.write("=" * 70 + "\n\n")
        for tc in test_data['test_cases']:
            f.write(f"[{tc['id']}] {tc['source']}\n")
            f.write(f"    Expected: {tc['target']}\n")
            f.write(f"    Errors: {', '.join(tc.get('errors', ['none']))}\n\n")
    print(f"Word/Google test file: {word_test_file}")


if __name__ == '__main__':
    main()
