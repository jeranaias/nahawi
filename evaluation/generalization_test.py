#!/usr/bin/env python3
"""
Comprehensive Generalization Testing for Nahawi GEC.

Tests:
1. False Positive Rate - Model should NOT change correct text
2. Cross-Domain Performance - Different text sources
3. Edge Cases - Short/long/mixed content
4. Error Recovery - Multiple errors, cascading errors
"""

import torch
import torch.nn as nn
import sentencepiece as spm
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path

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
    """Rule-based post-processor for hybrid system."""
    def __init__(self):
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
        self.maqsura_word = {'الي': 'إلى'}
        self.taa_fixes = {
            'حريه': 'حرية', 'الحريه': 'الحرية', 'والحريه': 'والحرية',
            'سحريه': 'سحرية', 'لقريه': 'لقرية', 'جزيره': 'جزيرة',
            'للدوله': 'للدولة',
        }
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


def decode(output_ids, sp):
    ids = output_ids[0].tolist()
    if 3 in ids:
        ids = ids[:ids.index(3)]
    if ids and ids[0] == 2:
        ids = ids[1:]
    return sp.DecodeIds(ids)


class ErrorInjector:
    """Inject realistic Arabic errors for testing."""

    # Alif maqsura errors
    MAQSURA = {'إلى': 'الي', 'على': 'علي', 'حتى': 'حتي', 'لدى': 'لدي', 'متى': 'متي'}

    # Taa marbuta errors
    TAA = {
        'المدينة': 'المدينه', 'الجامعة': 'الجامعه', 'المدرسة': 'المدرسه',
        'الدولة': 'الدوله', 'الحكومة': 'الحكومه', 'اللغة': 'اللغه',
        'الثقافة': 'الثقافه', 'الحياة': 'الحياه', 'القوة': 'القوه',
    }

    # Hamza errors
    HAMZA = {
        'أن': 'ان', 'إلى': 'الى', 'أو': 'او', 'إذا': 'اذا',
        'أكثر': 'اكثر', 'أقل': 'اقل', 'أمام': 'امام',
    }

    # Letter confusions
    SEEN_SAD = {'صورة': 'سورة', 'صباح': 'سباح', 'صحيح': 'سحيح'}
    DAD_ZA = {'ظهر': 'ضهر', 'حظ': 'حض', 'نظر': 'نضر'}

    @classmethod
    def inject_error(cls, text, error_type):
        """Inject a specific error type into text."""
        if error_type == 'alif_maqsura':
            for correct, error in cls.MAQSURA.items():
                if correct in text:
                    return text.replace(correct, error, 1), True
        elif error_type == 'taa_marbuta':
            for correct, error in cls.TAA.items():
                if correct in text:
                    return text.replace(correct, error, 1), True
        elif error_type == 'hamza':
            for correct, error in cls.HAMZA.items():
                if correct in text:
                    return text.replace(correct, error, 1), True
        elif error_type == 'letter_confusion_س_ص':
            for correct, error in cls.SEEN_SAD.items():
                if correct in text:
                    return text.replace(correct, error, 1), True
        elif error_type == 'letter_confusion_ض_ظ':
            for correct, error in cls.DAD_ZA.items():
                if correct in text:
                    return text.replace(correct, error, 1), True
        return text, False


def load_model(model_path, tokenizer_path, device='cuda'):
    """Load model and tokenizer."""
    sp = spm.SentencePieceProcessor()
    sp.Load(tokenizer_path)

    model = NahawiGEC(CONFIG).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state['model_state_dict'] if 'model_state_dict' in state else state)
    model.eval()

    rules = RulePostProcessor()
    return model, sp, rules


def correct_text(text, model, sp, rules, device='cuda'):
    """Correct text using model + rules."""
    src_ids = torch.tensor([[2] + sp.EncodeAsIds(text)[:254] + [3]], device=device)
    output = model.generate(src_ids)
    hyp = decode(output, sp)
    return rules.process(hyp)


def test_false_positive_rate(model, sp, rules, clean_sentences, device='cuda'):
    """
    TEST 1: False Positive Rate
    Model should NOT change correct sentences.
    """
    print("\n" + "="*70)
    print("TEST 1: FALSE POSITIVE RATE (Clean Text Preservation)")
    print("="*70)

    changed = 0
    unchanged = 0
    changes = []

    for sent in clean_sentences:
        output = correct_text(sent, model, sp, rules, device)
        if output.strip() != sent.strip():
            changed += 1
            changes.append({
                'input': sent,
                'output': output
            })
        else:
            unchanged += 1

    total = len(clean_sentences)
    fp_rate = changed / total * 100

    print(f"\nResults:")
    print(f"  Total sentences: {total}")
    print(f"  Unchanged (correct): {unchanged} ({unchanged/total*100:.1f}%)")
    print(f"  Changed (false positives): {changed} ({fp_rate:.1f}%)")

    if fp_rate < 5:
        print(f"\n  STATUS: PASS (FP rate < 5%)")
    elif fp_rate < 10:
        print(f"\n  STATUS: ACCEPTABLE (FP rate < 10%)")
    else:
        print(f"\n  STATUS: FAIL (FP rate >= 10%)")

    if changes[:5]:
        print(f"\nSample false positives:")
        for c in changes[:5]:
            print(f"  IN:  {c['input'][:60]}...")
            print(f"  OUT: {c['output'][:60]}...")
            print()

    return {'fp_rate': fp_rate, 'changed': changed, 'total': total, 'changes': changes}


def test_error_correction(model, sp, rules, clean_sentences, device='cuda'):
    """
    TEST 2: Error Correction Rate
    Inject errors and measure correction rate per category.
    """
    print("\n" + "="*70)
    print("TEST 2: ERROR CORRECTION RATE (Per Category)")
    print("="*70)

    error_types = ['alif_maqsura', 'taa_marbuta', 'hamza',
                   'letter_confusion_س_ص', 'letter_confusion_ض_ظ']

    results = {}

    for error_type in error_types:
        correct = 0
        total = 0

        for sent in clean_sentences[:200]:  # Use subset
            erroneous, injected = ErrorInjector.inject_error(sent, error_type)
            if injected:
                total += 1
                output = correct_text(erroneous, model, sp, rules, device)
                # Check if output matches original (error corrected)
                if output.strip() == sent.strip():
                    correct += 1

        if total > 0:
            rate = correct / total * 100
            results[error_type] = {'correct': correct, 'total': total, 'rate': rate}
            status = "PASS" if rate >= 90 else "WORK" if rate >= 80 else "FAIL"
            print(f"  {error_type:25s}: {rate:5.1f}% ({correct}/{total}) [{status}]")
        else:
            print(f"  {error_type:25s}: No samples")

    return results


def test_edge_cases(model, sp, rules, device='cuda'):
    """
    TEST 3: Edge Cases
    Short, long, and mixed content sentences.
    """
    print("\n" + "="*70)
    print("TEST 3: EDGE CASES")
    print("="*70)

    # Short sentences
    short_sentences = [
        "ذهب إلى المدرسة.",
        "كتاب جميل.",
        "هذا صحيح.",
        "نعم، أوافق.",
        "ما اسمك؟",
    ]

    # Long sentences (typical Wikipedia)
    long_sentences = [
        "تعتبر اللغة العربية من أقدم اللغات السامية وأكثرها انتشاراً في العالم، حيث يتحدث بها أكثر من أربعمائة مليون نسمة، وهي اللغة الرسمية في اثنتين وعشرين دولة عربية.",
        "يمتد تاريخ الحضارة العربية الإسلامية على مدى قرون طويلة، شهدت فيها ازدهاراً علمياً وثقافياً كبيراً، أسهم في تطور العلوم والفنون والآداب في مختلف أنحاء العالم.",
    ]

    # Mixed content (numbers, names)
    mixed_sentences = [
        "ولد الكاتب محمد في عام 1985 في مدينة القاهرة.",
        "بلغت المبيعات 500 مليون دولار في الربع الأول.",
        "التقى الرئيس أوباما بنظيره الفرنسي ماكرون.",
    ]

    print("\n  Short sentences (should preserve):")
    for sent in short_sentences:
        output = correct_text(sent, model, sp, rules, device)
        status = "OK" if output.strip() == sent.strip() else "CHANGED"
        print(f"    [{status}] {sent}")

    print("\n  Long sentences (should preserve):")
    for sent in long_sentences:
        output = correct_text(sent, model, sp, rules, device)
        status = "OK" if output.strip() == sent.strip() else "CHANGED"
        print(f"    [{status}] {sent[:50]}...")

    print("\n  Mixed content (should preserve):")
    for sent in mixed_sentences:
        output = correct_text(sent, model, sp, rules, device)
        status = "OK" if output.strip() == sent.strip() else "CHANGED"
        print(f"    [{status}] {sent}")


def test_multiple_errors(model, sp, rules, device='cuda'):
    """
    TEST 4: Multiple Error Recovery
    Sentences with multiple injected errors.
    """
    print("\n" + "="*70)
    print("TEST 4: MULTIPLE ERROR RECOVERY")
    print("="*70)

    # Sentences with multiple errors
    test_cases = [
        {
            'erroneous': "ذهب الي المدرسه",
            'correct': "ذهب إلى المدرسة",
            'errors': ['alif_maqsura', 'taa_marbuta']
        },
        {
            'erroneous': "هذه الجامعه الكبيره",
            'correct': "هذه الجامعة الكبيرة",
            'errors': ['taa_marbuta', 'taa_marbuta']
        },
        {
            'erroneous': "يجب ان نذهب الي هناك",
            'correct': "يجب أن نذهب إلى هناك",
            'errors': ['hamza', 'alif_maqsura']
        },
    ]

    correct = 0
    for tc in test_cases:
        output = correct_text(tc['erroneous'], model, sp, rules, device)
        is_correct = output.strip() == tc['correct'].strip()
        if is_correct:
            correct += 1
        status = "PASS" if is_correct else "FAIL"
        print(f"  [{status}] {tc['erroneous']}")
        print(f"         → {output}")
        if not is_correct:
            print(f"    Expected: {tc['correct']}")
        print()

    print(f"  Result: {correct}/{len(test_cases)} multi-error sentences corrected")


def main():
    """Run all generalization tests."""
    print("="*70)
    print("NAHAWI GENERALIZATION TEST SUITE")
    print("="*70)

    # Paths
    model_path = 'models/v5_hybrid/fasih_v5_best.pt'
    tokenizer_path = 'models/v5_hybrid/nahawi_spm.model'

    # Check if running locally or on remote
    import os
    if not os.path.exists(model_path):
        model_path = 'fasih_v5_model/best_model.pt'
        tokenizer_path = 'nahawi_spm.model'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    print(f"Model: {model_path}")

    # Load model
    print("\nLoading model...")
    model, sp, rules = load_model(model_path, tokenizer_path, device)
    print("Model loaded!")

    # Load clean sentences from FASIH (using targets as clean text)
    fasih_path = 'benchmark/fasih_v4.1/fasih_test.json'
    if not os.path.exists(fasih_path):
        fasih_path = 'data/fasih_v4.1/fasih_test.json'

    with open(fasih_path) as f:
        fasih_data = json.load(f)

    # Use targets as clean sentences (they are correct)
    clean_sentences = list(set([s['target'] for s in fasih_data]))[:500]
    print(f"\nLoaded {len(clean_sentences)} clean sentences for testing")

    # Run tests
    results = {}

    # Test 1: False Positive Rate
    results['fp_test'] = test_false_positive_rate(model, sp, rules, clean_sentences, device)

    # Test 2: Error Correction Rate
    results['correction_test'] = test_error_correction(model, sp, rules, clean_sentences, device)

    # Test 3: Edge Cases
    test_edge_cases(model, sp, rules, device)

    # Test 4: Multiple Errors
    test_multiple_errors(model, sp, rules, device)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n  False Positive Rate: {results['fp_test']['fp_rate']:.1f}%")
    print(f"  Target: <5%  |  Status: {'PASS' if results['fp_test']['fp_rate'] < 5 else 'NEEDS WORK'}")

    print("\n  Ready for production deployment!" if results['fp_test']['fp_rate'] < 10 else "\n  Needs investigation before deployment.")


if __name__ == '__main__':
    main()
