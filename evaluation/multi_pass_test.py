#!/usr/bin/env python3
"""
Multi-Pass Correction Test for Heavily Corrupted Text.
Tests whether running the model multiple times improves correction of sentences with many errors.
"""

import torch
import torch.nn as nn
import sentencepiece as spm
import math

# Model config
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
        self.masc_nouns = {'قمر', 'متحف', 'هيكل', 'تحيز', 'إثبات', 'معلم', 'عدد', 'نشاط', 'اكتشاف', 'مبنى', 'نظام', 'برنامج', 'مشروع', 'تطور', 'تأثير', 'تغيير', 'تقدم', 'إنجاز', 'حدث', 'موقع', 'مكان', 'فندق', 'قصر', 'معبد', 'مسجد', 'بيت', 'منزل', 'طريق', 'شارع', 'نهر', 'بحر', 'جبل', 'وادي', 'سهل', 'حقل', 'ملعب', 'مطار', 'بشكل', 'هو'}
        self.adj_masc = {'صغيرة': 'صغير', 'كبيرة': 'كبير', 'طويلة': 'طويل', 'قصيرة': 'قصير', 'جديدة': 'جديد', 'قديمة': 'قديم', 'جميلة': 'جميل', 'كاملة': 'كامل', 'علمية': 'علمي', 'عسكرية': 'عسكري', 'سياسية': 'سياسي', 'رئيسية': 'رئيسي', 'أساسية': 'أساسي', 'خاصة': 'خاص', 'عامة': 'عام', 'سريعة': 'سريع', 'رابعة': 'رابع', 'ثالثة': 'ثالث', 'ثانية': 'ثاني', 'أولى': 'أول', 'سعيدة': 'سعيد', 'واسعة': 'واسع'}
        self.maqsura_word = {'الي': 'إلى'}
        self.taa_fixes = {'حريه': 'حرية', 'الحريه': 'الحرية', 'والحريه': 'والحرية', 'سحريه': 'سحرية', 'لقريه': 'لقرية', 'جزيره': 'جزيرة', 'للدوله': 'للدولة'}
        self.prep_fixes = [('يعتمد عن', 'يعتمد على')]

    def process(self, text):
        for wrong, correct in self.prep_fixes:
            text = text.replace(wrong, correct)
        words = text.split()
        result = words.copy()
        for i in range(len(words)):
            word = words[i]
            clean = word.rstrip('،.؟!؛:"')
            punct = word[len(clean):]
            if clean in self.maqsura_word:
                result[i] = self.maqsura_word[clean] + punct
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


def load_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sp = spm.SentencePieceProcessor()
    sp.Load('nahawi_spm.model')

    model = NahawiGEC(CONFIG).to(device)
    state = torch.load('fasih_v5_model/best_model.pt', map_location=device)
    model.load_state_dict(state['model_state_dict'] if 'model_state_dict' in state else state)
    model.eval()
    rules = RulePostProcessor()
    return model, sp, rules, device


def correct(text, model, sp, rules, device):
    src_ids = torch.tensor([[2] + sp.EncodeAsIds(text)[:254] + [3]], device=device)
    output = model.generate(src_ids)
    ids = output[0].tolist()
    if 3 in ids: ids = ids[:ids.index(3)]
    if ids and ids[0] == 2: ids = ids[1:]
    return rules.process(sp.DecodeIds(ids))


def multi_pass(text, model, sp, rules, device, max_passes=5):
    """Run correction multiple times until convergence."""
    current = text
    history = [text]
    for i in range(max_passes):
        result = correct(current, model, sp, rules, device)
        if result == current:  # Converged
            break
        current = result
        history.append(result)
    return history


def count_errors(corrupted, correct):
    """Simple error count based on word differences."""
    c_words = corrupted.split()
    r_words = correct.split()
    errors = 0
    for cw, rw in zip(c_words, r_words):
        if cw != rw:
            errors += 1
    return errors


def main():
    print("Loading model...")
    model, sp, rules, device = load_model()
    print(f"Model loaded on {device}")

    # Test cases with multiple errors
    test_cases = [
        {
            'corrupted': 'ذهب الي المدرسه الكبيره',
            'correct': 'ذهب إلى المدرسة الكبيرة',
            'errors': 3
        },
        {
            'corrupted': 'هذه الجامعه الكبيره في المدينه',
            'correct': 'هذه الجامعة الكبيرة في المدينة',
            'errors': 3
        },
        {
            'corrupted': 'يجب ان نذهب الي هناك حتي نري الحقيقه',
            'correct': 'يجب أن نذهب إلى هناك حتى نرى الحقيقة',
            'errors': 5
        },
        {
            'corrupted': 'الطالب ذهب الي الجامعه لكي يتعلم اللغه العربيه',
            'correct': 'الطالب ذهب إلى الجامعة لكي يتعلم اللغة العربية',
            'errors': 4
        },
        {
            'corrupted': 'ان الحريه والعداله هما اساس الديمقراطيه',
            'correct': 'إن الحرية والعدالة هما أساس الديمقراطية',
            'errors': 5
        },
        {
            'corrupted': 'المدرسه الجديده في المدينه الكبيره جميله جدا',
            'correct': 'المدرسة الجديدة في المدينة الكبيرة جميلة جدا',
            'errors': 5
        },
        {
            'corrupted': 'سافر الي مصر ثم الي لبنان ثم الي الاردن',
            'correct': 'سافر إلى مصر ثم إلى لبنان ثم إلى الأردن',
            'errors': 4
        },
    ]

    print('=' * 70)
    print('MULTI-PASS CORRECTION TEST')
    print('=' * 70)

    results = {'full': 0, 'partial': 0, 'total': len(test_cases)}
    pass_counts = []

    for tc in test_cases:
        print(f"\nInput ({tc['errors']} errors): {tc['corrupted']}")
        print(f"Expected:              {tc['correct']}")

        history = multi_pass(tc['corrupted'], model, sp, rules, device)

        for i, h in enumerate(history):
            if i == 0:
                continue
            marker = '✓' if h == tc['correct'] else '○'
            print(f"  Pass {i}: {marker} {h}")

        final = history[-1]
        passes_needed = len(history) - 1

        if final == tc['correct']:
            print(f"  → FULLY CORRECTED in {passes_needed} pass(es)")
            results['full'] += 1
            pass_counts.append(passes_needed)
        else:
            remaining = count_errors(final, tc['correct'])
            print(f"  → PARTIAL ({remaining} errors remain after {passes_needed} passes)")
            results['partial'] += 1

    print('\n' + '=' * 70)
    print('SUMMARY')
    print('=' * 70)
    print(f"\nFully corrected: {results['full']}/{results['total']} ({results['full']/results['total']*100:.0f}%)")
    print(f"Partial:         {results['partial']}/{results['total']} ({results['partial']/results['total']*100:.0f}%)")

    if pass_counts:
        print(f"\nAverage passes for full correction: {sum(pass_counts)/len(pass_counts):.1f}")
        print(f"Max passes needed: {max(pass_counts)}")

    print('\n' + '=' * 70)
    print('CONCLUSION')
    print('=' * 70)
    if results['full'] / results['total'] >= 0.8:
        print("Multi-pass significantly improves multi-error correction!")
    elif results['full'] / results['total'] >= 0.5:
        print("Multi-pass provides moderate improvement for multi-error cases.")
    else:
        print("Multi-pass has limited benefit - model converges quickly.")


if __name__ == '__main__':
    main()
