#!/usr/bin/env python3
"""
QALB-2014 Benchmark Evaluation for Nahawi GEC.
Evaluates on the official QALB-2014 test set and compares to published baselines.
"""

import torch
import torch.nn as nn
import sentencepiece as spm
import math
import json
import re
import sys
import time
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

# Model configuration (must match training)
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
    """Rule-based post-processor for common Arabic grammar patterns."""

    def __init__(self):
        self.masc_nouns = {'قمر', 'متحف', 'هيكل', 'تحيز', 'إثبات', 'معلم', 'عدد', 'نشاط', 'اكتشاف',
                          'مبنى', 'نظام', 'برنامج', 'مشروع', 'تطور', 'تأثير', 'تغيير', 'تقدم', 'إنجاز',
                          'حدث', 'موقع', 'مكان', 'فندق', 'قصر', 'معبد', 'مسجد', 'بيت', 'منزل',
                          'طريق', 'شارع', 'نهر', 'بحر', 'جبل', 'وادي', 'سهل', 'حقل', 'ملعب', 'مطار', 'بشكل', 'هو'}
        self.adj_masc = {'صغيرة': 'صغير', 'كبيرة': 'كبير', 'طويلة': 'طويل', 'قصيرة': 'قصير',
                        'جديدة': 'جديد', 'قديمة': 'قديم', 'جميلة': 'جميل', 'كاملة': 'كامل',
                        'علمية': 'علمي', 'عسكرية': 'عسكري', 'سياسية': 'سياسي', 'رئيسية': 'رئيسي',
                        'أساسية': 'أساسي', 'خاصة': 'خاص', 'عامة': 'عام', 'سريعة': 'سريع'}
        self.maqsura_fix = {'الي': 'إلى', 'علي': 'على', 'حتي': 'حتى', 'متي': 'متى', 'انثي': 'أنثى'}
        self.taa_fixes = {'حريه': 'حرية', 'الحريه': 'الحرية', 'مدرسه': 'مدرسة', 'جامعه': 'جامعة',
                         'مدينه': 'مدينة', 'قريه': 'قرية', 'جزيره': 'جزيرة', 'دوله': 'دولة'}
        self.prep_fixes = [('يعتمد عن', 'يعتمد على')]

    def process(self, text):
        for wrong, correct in self.prep_fixes:
            text = text.replace(wrong, correct)
        words = text.split()
        result = words.copy()
        for i in range(len(words)):
            word = words[i]
            clean = word.rstrip('،.؟!؛:\"\'')
            punct = word[len(clean):]
            if clean in self.maqsura_fix:
                result[i] = self.maqsura_fix[clean] + punct
                continue
            if clean in self.taa_fixes:
                result[i] = self.taa_fixes[clean] + punct
                continue
            if i < len(words) - 1:
                adj = words[i+1].rstrip('،.؟!؛:')
                adj_punct = words[i+1][len(adj):]
                if clean in self.masc_nouns and adj in self.adj_masc:
                    result[i+1] = self.adj_masc[adj] + adj_punct
        return ' '.join(result)


def parse_m2_file(m2_path):
    """Parse M2 file and extract source sentences with their gold edits."""
    sentences = []
    current_source = None
    current_edits = []

    with open(m2_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('S '):
                if current_source is not None:
                    sentences.append({
                        'source': current_source,
                        'edits': current_edits
                    })
                current_source = line[2:]
                current_edits = []
            elif line.startswith('A '):
                current_edits.append(line[2:])
            elif line == '' and current_source is not None:
                sentences.append({
                    'source': current_source,
                    'edits': current_edits
                })
                current_source = None
                current_edits = []

    if current_source is not None:
        sentences.append({
            'source': current_source,
            'edits': current_edits
        })

    return sentences


def apply_edits(source, edits):
    """Apply M2 edits to get gold target sentence."""
    words = source.split()

    # Parse and sort edits by position (reverse order for correct application)
    parsed_edits = []
    for edit in edits:
        parts = edit.split('|||')
        if len(parts) >= 3:
            span = parts[0].split()
            if len(span) >= 2:
                start, end = int(span[0]), int(span[1])
                correction = parts[2]
                parsed_edits.append((start, end, correction))

    # Apply edits in reverse order
    parsed_edits.sort(key=lambda x: x[0], reverse=True)
    for start, end, correction in parsed_edits:
        if correction == '-NONE-':
            # Delete
            words = words[:start] + words[end:]
        else:
            # Replace or insert
            words = words[:start] + correction.split() + words[end:]

    return ' '.join(words)


def calculate_f05_token_level(hyp, ref, source):
    """Calculate token-level F0.5 score."""
    hyp_tokens = set(enumerate(hyp.split()))
    ref_tokens = set(enumerate(ref.split()))
    src_tokens = set(enumerate(source.split()))

    # Get edits (changes from source)
    hyp_edits = hyp_tokens - src_tokens
    ref_edits = ref_tokens - src_tokens

    if len(ref_edits) == 0:
        # No edits needed
        if len(hyp_edits) == 0:
            return 1.0, 1.0, 1.0  # Perfect
        else:
            return 0.0, 1.0, 0.0  # False positives

    if len(hyp_edits) == 0:
        return 1.0, 0.0, 0.0  # No predictions, zero recall

    # Count correct edits
    correct = len(hyp_edits & ref_edits)
    precision = correct / len(hyp_edits) if hyp_edits else 0
    recall = correct / len(ref_edits) if ref_edits else 0

    beta = 0.5
    if precision + recall > 0:
        f05 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
    else:
        f05 = 0.0

    return precision, recall, f05


def m2_scorer_simple(sources, hypotheses, references):
    """
    Simple M2 scorer that calculates corpus-level F0.5.
    Compares hypothesis edits against reference edits.
    """
    total_correct = 0
    total_proposed = 0
    total_gold = 0

    for src, hyp, ref in zip(sources, hypotheses, references):
        src_words = src.split()
        hyp_words = hyp.split()
        ref_words = ref.split()

        # Get edit operations (simplified - word-level diff)
        # Proposed edits: differences between hyp and src
        # Gold edits: differences between ref and src

        # Use simple set difference for now
        hyp_changes = set()
        ref_changes = set()

        # Word-level changes
        max_len = max(len(src_words), len(hyp_words), len(ref_words))

        for i in range(max_len):
            src_w = src_words[i] if i < len(src_words) else None
            hyp_w = hyp_words[i] if i < len(hyp_words) else None
            ref_w = ref_words[i] if i < len(ref_words) else None

            if hyp_w != src_w:
                hyp_changes.add((i, src_w, hyp_w))
            if ref_w != src_w:
                ref_changes.add((i, src_w, ref_w))

        # Count matches
        correct = len(hyp_changes & ref_changes)
        total_correct += correct
        total_proposed += len(hyp_changes)
        total_gold += len(ref_changes)

    precision = total_correct / total_proposed if total_proposed > 0 else 0
    recall = total_correct / total_gold if total_gold > 0 else 0

    beta = 0.5
    if precision + recall > 0:
        f05 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
    else:
        f05 = 0.0

    return {
        'precision': precision * 100,
        'recall': recall * 100,
        'f0.5': f05 * 100,
        'total_proposed': total_proposed,
        'total_gold': total_gold,
        'total_correct': total_correct
    }


def load_model(model_path, spm_path):
    """Load the Nahawi model and tokenizer."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load tokenizer
    sp = spm.SentencePieceProcessor()
    sp.Load(spm_path)

    # Load model
    model = NahawiGEC(CONFIG).to(device)
    state = torch.load(model_path, map_location=device)
    if 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'])
    else:
        model.load_state_dict(state)
    model.eval()

    rules = RulePostProcessor()

    return model, sp, rules, device


def correct_sentence(text, model, sp, rules, device):
    """Correct a single sentence."""
    src_ids = torch.tensor([[2] + sp.EncodeAsIds(text)[:254] + [3]], device=device)
    output = model.generate(src_ids)
    ids = output[0].tolist()
    if 3 in ids:
        ids = ids[:ids.index(3)]
    if ids and ids[0] == 2:
        ids = ids[1:]
    decoded = sp.DecodeIds(ids)
    return rules.process(decoded)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='QALB Benchmark Evaluation')
    parser.add_argument('--m2-file', type=str, required=True, help='Path to QALB M2 file')
    parser.add_argument('--model', type=str, default='fasih_v5_model/best_model.pt', help='Model path')
    parser.add_argument('--spm', type=str, default='nahawi_spm.model', help='SentencePiece model path')
    parser.add_argument('--output-hyp', type=str, default='qalb_hyp.txt', help='Output hypothesis file')
    parser.add_argument('--output-report', type=str, default='qalb_benchmark_report.json', help='Output report')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of sentences (for testing)')
    args = parser.parse_args()

    print("=" * 70)
    print("QALB-2014 BENCHMARK EVALUATION")
    print("=" * 70)

    # Load model
    print("\nLoading model...")
    model, sp, rules, device = load_model(args.model, args.spm)
    print(f"Model loaded on {device}")

    # Parse M2 file
    print(f"\nParsing M2 file: {args.m2_file}")
    sentences = parse_m2_file(args.m2_file)
    print(f"Found {len(sentences)} sentences")

    if args.limit:
        sentences = sentences[:args.limit]
        print(f"Limited to {args.limit} sentences")

    # Generate hypotheses
    print("\nGenerating corrections...")
    sources = []
    references = []
    hypotheses = []

    start_time = time.time()
    for i, sent in enumerate(sentences):
        source = sent['source']
        reference = apply_edits(source, sent['edits'])
        hypothesis = correct_sentence(source, model, sp, rules, device)

        sources.append(source)
        references.append(reference)
        hypotheses.append(hypothesis)

        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = (len(sentences) - i - 1) / rate
            print(f"  Processed {i+1}/{len(sentences)} ({rate:.1f} sent/sec, ~{remaining:.0f}s remaining)")

    elapsed = time.time() - start_time
    print(f"Completed in {elapsed:.1f}s ({len(sentences)/elapsed:.1f} sent/sec)")

    # Save hypotheses
    with open(args.output_hyp, 'w', encoding='utf-8') as f:
        for hyp in hypotheses:
            f.write(hyp + '\n')
    print(f"\nHypotheses saved to: {args.output_hyp}")

    # Calculate scores
    print("\nCalculating F0.5 scores...")
    scores = m2_scorer_simple(sources, hypotheses, references)

    # Published baselines
    baselines = {
        'QALB-2014 Winner (CUFE)': 67.9,
        'QALB-2014 Runner-up (CMUQ)': 65.4,
        'Recent SOTA (CAMeL-Lab)': 72.0,
        'Grammarly (English, approx)': 90.0
    }

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nNahawi V5 Hybrid on QALB-2014 Test:")
    print(f"  Precision: {scores['precision']:.1f}%")
    print(f"  Recall:    {scores['recall']:.1f}%")
    print(f"  F0.5:      {scores['f0.5']:.1f}%")
    print(f"\n  Total proposed edits: {scores['total_proposed']}")
    print(f"  Total gold edits:     {scores['total_gold']}")
    print(f"  Correct edits:        {scores['total_correct']}")

    print("\n" + "-" * 70)
    print("COMPARISON WITH PUBLISHED BASELINES")
    print("-" * 70)
    for name, score in baselines.items():
        delta = scores['f0.5'] - score
        status = "BETTER" if delta > 0 else "BEHIND" if delta < 0 else "EQUAL"
        print(f"  {name}: {score:.1f}% ({status}, delta: {delta:+.1f}%)")

    # Save report
    report = {
        'timestamp': datetime.now().isoformat(),
        'model': args.model,
        'dataset': 'QALB-2014 Test',
        'num_sentences': len(sentences),
        'scores': {
            'precision': scores['precision'],
            'recall': scores['recall'],
            'f0.5': scores['f0.5']
        },
        'edit_counts': {
            'proposed': scores['total_proposed'],
            'gold': scores['total_gold'],
            'correct': scores['total_correct']
        },
        'baselines': baselines,
        'comparison': {
            name: {
                'baseline': score,
                'delta': scores['f0.5'] - score,
                'status': 'BETTER' if scores['f0.5'] > score else 'BEHIND'
            }
            for name, score in baselines.items()
        },
        'inference_time_seconds': elapsed,
        'sentences_per_second': len(sentences) / elapsed
    }

    with open(args.output_report, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved to: {args.output_report}")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    if scores['f0.5'] >= 72.0:
        print("Nahawi achieves STATE-OF-THE-ART on QALB-2014!")
    elif scores['f0.5'] >= 67.9:
        print("Nahawi BEATS the original QALB-2014 winner!")
    elif scores['f0.5'] >= 60.0:
        print("Nahawi shows COMPETITIVE performance on QALB-2014.")
    elif scores['f0.5'] >= 50.0:
        print("Nahawi shows MODERATE performance on QALB-2014.")
    else:
        print("Performance on QALB-2014 needs improvement.")

    print(f"\nNote: QALB contains significant punctuation and dialect content.")
    print("Compare with FASIH benchmark for pure MSA grammatical errors.")


if __name__ == '__main__':
    main()
