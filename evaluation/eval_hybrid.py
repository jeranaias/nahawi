#!/usr/bin/env python3
"""Evaluate V5 model + Rule post-processor hybrid."""
import torch
import torch.nn as nn
import sentencepiece as spm
import json
import math
import re
from collections import defaultdict

class RulePostProcessor:
    """Rules for gender agreement, alif maqsura, and taa marbuta."""
    def __init__(self):
        # Common masculine nouns/patterns that require masculine adjectives
        self.masc_nouns = {
            'قمر', 'متحف', 'هيكل', 'تحيز', 'إثبات', 'معلم', 'عدد',
            'نشاط', 'اكتشاف', 'مبنى', 'نظام', 'برنامج', 'مشروع', 'تطور',
            'تأثير', 'تغيير', 'تقدم', 'إنجاز', 'حدث', 'موقع', 'مكان',
            'فندق', 'قصر', 'معبد', 'مسجد', 'بيت', 'منزل', 'طريق', 'شارع',
            'نهر', 'بحر', 'جبل', 'وادي', 'سهل', 'حقل', 'ملعب', 'مطار',
            'بشكل', 'هو',
        }
        self.adj_masc = {
            'صغيرة': 'صغير', 'كبيرة': 'كبير', 'طويلة': 'طويل', 'قصيرة': 'قصير',
            'جديدة': 'جديد', 'قديمة': 'قديم', 'جميلة': 'جميل', 'كاملة': 'كامل',
            'علمية': 'علمي', 'عسكرية': 'عسكري', 'سياسية': 'سياسي', 'رئيسية': 'رئيسي',
            'أساسية': 'أساسي', 'خاصة': 'خاص', 'عامة': 'عام', 'سريعة': 'سريع',
            'رابعة': 'رابع', 'ثالثة': 'ثالث', 'ثانية': 'ثاني', 'أولى': 'أول',
            'سعيدة': 'سعيد', 'واسعة': 'واسع',
        }

        # Alif maqsura: الي → إلى (very common error!)
        self.maqsura_word = {'الي': 'إلى'}

        # Taa marbuta fixes (ه → ة)
        self.taa_fixes = {
            'حريه': 'حرية', 'الحريه': 'الحرية', 'والحريه': 'والحرية',
            'سحريه': 'سحرية', 'لقريه': 'لقرية', 'جزيره': 'جزيرة',
            'للدوله': 'للدولة',
        }

        # Wrong preposition fixes
        self.prep_fixes = [
            ('يعتمد عن', 'يعتمد على'),
        ]

    def process(self, text):
        # 0. Wrong preposition fixes (phrase-level)
        for wrong, correct in self.prep_fixes:
            text = text.replace(wrong, correct)

        words = text.split()
        result = words.copy()

        for i in range(len(words)):
            word = words[i]
            clean = word.rstrip('،.؟!؛:"')
            punct = word[len(clean):]

            # 1. Alif maqsura: الي → إلى
            if clean in self.maqsura_word:
                result[i] = self.maqsura_word[clean] + punct
                continue

            # 2. Taa marbuta fixes
            if clean in self.taa_fixes:
                result[i] = self.taa_fixes[clean] + punct
                continue

            # 3. Gender agreement (noun + adj)
            if i < len(words) - 1:
                adj = words[i+1].rstrip('،.؟!؛:')
                adj_punct = words[i+1][len(adj):]
                if clean in self.masc_nouns and adj in self.adj_masc:
                    result[i+1] = self.adj_masc[adj] + adj_punct

        return ' '.join(result)

CONFIG = {'vocab_size': 32000, 'd_model': 768, 'nhead': 12, 'num_encoder_layers': 6,
          'num_decoder_layers': 6, 'dim_feedforward': 3072, 'dropout': 0.1, 'max_seq_len': 256}

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:, :x.size(1)]

class NahawiGEC(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config['vocab_size'], config['d_model'])
        self.pos_encoder = PositionalEncoding(config['d_model'], config['max_seq_len'])
        encoder_layer = nn.TransformerEncoderLayer(d_model=config['d_model'], nhead=config['nhead'],
            dim_feedforward=config['dim_feedforward'], dropout=config['dropout'], batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config['num_encoder_layers'])
        decoder_layer = nn.TransformerDecoderLayer(d_model=config['d_model'], nhead=config['nhead'],
            dim_feedforward=config['dim_feedforward'], dropout=config['dropout'], batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config['num_decoder_layers'])
        self.output_projection = nn.Linear(config['d_model'], config['vocab_size'])
        self.output_projection.weight = self.embedding.weight
        self.d_model = config['d_model']

    @torch.no_grad()
    def generate(self, src_ids, max_len=256, eos_id=3):
        self.eval(); device = src_ids.device
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
            if next_token.item() == eos_id: break
        return generated

def decode(output_ids, sp):
    ids = output_ids[0].tolist()
    if 3 in ids: ids = ids[:ids.index(3)]
    if ids and ids[0] == 2: ids = ids[1:]
    return sp.DecodeIds(ids)

def main():
    print('='*60)
    print('V5 + RULES HYBRID EVALUATION')
    print('='*60)

    device = 'cuda'
    sp = spm.SentencePieceProcessor()
    sp.Load('nahawi_spm.model')
    pp = RulePostProcessor()

    with open('data/fasih_v4.1/fasih_test.json') as f:
        test_data = json.load(f)
    print(f'Test samples: {len(test_data)}')

    model = NahawiGEC(CONFIG).to(device)
    state = torch.load('fasih_v5_model/best_model.pt', map_location=device)
    model.load_state_dict(state['model_state_dict'] if 'model_state_dict' in state else state)
    model.eval()

    type_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})

    for i, sample in enumerate(test_data):
        src, tgt = sample['source'], sample['target']
        error_type = sample.get('error_type', 'unknown')

        src_ids = torch.tensor([[2] + sp.EncodeAsIds(src)[:254] + [3]], device=device)
        hyp = decode(model.generate(src_ids), sp)
        hyp = pp.process(hyp)  # Apply rules!

        src_w, hyp_w, tgt_w = src.split(), hyp.split(), tgt.split()
        min_len = min(len(src_w), len(tgt_w), len(hyp_w))
        for j in range(min_len):
            if src_w[j] != tgt_w[j]:
                if hyp_w[j] == tgt_w[j]:
                    type_stats[error_type]['tp'] += 1
                else:
                    type_stats[error_type]['fn'] += 1
            elif hyp_w[j] != src_w[j]:
                type_stats[error_type]['fp'] += 1

        if (i+1) % 400 == 0:
            print(f'  {i+1}/{len(test_data)}...')

    print()
    print('='*60)
    print('V5 + RULES RESULTS')
    print('='*60)

    overall_tp, overall_fp, overall_fn = 0, 0, 0
    pass_count = 0

    for error_type in sorted(type_stats.keys()):
        stats = type_stats[error_type]
        tp, fp, fn = stats['tp'], stats['fp'], stats['fn']
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f05 = (1.25 * p * r) / (0.25 * p + r) if (p + r) > 0 else 0
        status = 'PASS' if f05 >= 0.9 else 'WORK' if f05 >= 0.7 else 'FAIL'
        if f05 >= 0.9:
            pass_count += 1
        print(f'{error_type:30s}: F0.5={f05*100:5.1f}% [{status}]')
        overall_tp += tp
        overall_fp += fp
        overall_fn += fn

    p = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0
    r = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0
    f05 = (1.25 * p * r) / (0.25 * p + r) if (p + r) > 0 else 0

    print('='*60)
    print(f'OVERALL F0.5: {f05*100:.1f}%')
    print(f'PASS categories: {pass_count}/13')
    print('='*60)

    if pass_count == 13:
        print('*** ALL CATEGORIES PASS! ARABIC GRAMMARLY ACHIEVED! ***')
    elif pass_count >= 11:
        print('EXCELLENT - Almost there!')

if __name__ == '__main__':
    main()
