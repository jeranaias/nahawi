#!/usr/bin/env python3
"""
Nahawi Scientific Evaluation Suite

Implements the rigorous evaluation framework for validating
production-readiness of the Arabic GEC system.

Usage:
    python scientific_evaluation.py --all           # Run all tests
    python scientific_evaluation.py --benchmark     # FASIH benchmark only
    python scientific_evaluation.py --fp-test       # False positive test
    python scientific_evaluation.py --cross-domain  # Cross-domain test
    python scientific_evaluation.py --robustness    # Robustness tests
    python scientific_evaluation.py --ablation      # Ablation studies
"""

import torch
import torch.nn as nn
import sentencepiece as spm
import json
import math
import random
import argparse
import numpy as np
from collections import defaultdict
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Model configuration
CONFIG = {
    'vocab_size': 32000, 'd_model': 768, 'nhead': 12,
    'num_encoder_layers': 6, 'num_decoder_layers': 6,
    'dim_feedforward': 3072, 'dropout': 0.1, 'max_seq_len': 256
}


# ============================================================================
# MODEL CLASSES
# ============================================================================

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
            'قمر', 'متحف', 'هيكل', 'تحيز', 'إثبات', 'معلم', 'عدد', 'نشاط',
            'اكتشاف', 'مبنى', 'نظام', 'برنامج', 'مشروع', 'تطور', 'تأثير',
            'تغيير', 'تقدم', 'إنجاز', 'حدث', 'موقع', 'مكان', 'فندق', 'قصر',
            'معبد', 'مسجد', 'بيت', 'منزل', 'طريق', 'شارع', 'نهر', 'بحر',
            'جبل', 'وادي', 'سهل', 'حقل', 'ملعب', 'مطار', 'بشكل', 'هو',
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
            'سحريه': 'سحرية', 'لقريه': 'لقرية', 'جزيره': 'جزيرة', 'للدوله': 'للدولة',
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
                adj = words[i + 1].rstrip('،.؟!؛:')
                adj_punct = words[i + 1][len(adj):]
                if clean in self.masc_nouns and adj in self.adj_masc:
                    result[i + 1] = self.adj_masc[adj] + adj_punct
        return ' '.join(result)


# ============================================================================
# ERROR INJECTION
# ============================================================================

class ErrorInjector:
    """Controlled error injection for testing."""

    HAMZA = {
        'أن': 'ان', 'إلى': 'الى', 'أو': 'او', 'إذا': 'اذا',
        'أكثر': 'اكثر', 'أقل': 'اقل', 'أمام': 'امام', 'إن': 'ان',
        'أي': 'اي', 'إنه': 'انه', 'أنه': 'انه',
    }

    ALIF_MAQSURA = {
        'إلى': 'الي', 'على': 'علي', 'حتى': 'حتي', 'لدى': 'لدي',
        'متى': 'متي', 'مدى': 'مدي', 'سوى': 'سوي',
    }

    TAA_MARBUTA = {
        'المدينة': 'المدينه', 'الجامعة': 'الجامعه', 'المدرسة': 'المدرسه',
        'الدولة': 'الدوله', 'الحكومة': 'الحكومه', 'اللغة': 'اللغه',
        'الثقافة': 'الثقافه', 'الحياة': 'الحياه', 'القوة': 'القوه',
        'الكبيرة': 'الكبيره', 'الصغيرة': 'الصغيره', 'الجديدة': 'الجديده',
        'القديمة': 'القديمه', 'العربية': 'العربيه', 'المصرية': 'المصريه',
    }

    DEFINITENESS = {
        'الكتاب الجديد': 'الكتاب جديد',
        'المدينة الكبيرة': 'المدينة كبيرة',
    }

    @classmethod
    def inject(cls, text: str, error_type: str) -> Tuple[str, bool]:
        """Inject a specific error type. Returns (text, was_injected)."""
        mappings = {
            'hamza': cls.HAMZA,
            'alif_maqsura': cls.ALIF_MAQSURA,
            'taa_marbuta': cls.TAA_MARBUTA,
            'definiteness': cls.DEFINITENESS,
        }

        if error_type not in mappings:
            return text, False

        for correct, error in mappings[error_type].items():
            if correct in text:
                return text.replace(correct, error, 1), True

        return text, False

    @classmethod
    def inject_multiple(cls, text: str, error_types: List[str]) -> Tuple[str, List[str]]:
        """Inject multiple error types. Returns (text, list_of_injected_types)."""
        injected = []
        current = text
        for error_type in error_types:
            result, was_injected = cls.inject(current, error_type)
            if was_injected:
                current = result
                injected.append(error_type)
        return current, injected


# ============================================================================
# METRICS
# ============================================================================

def calculate_f05(tp: int, fp: int, fn: int) -> float:
    """Calculate F0.5 score."""
    if tp + fp == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)

    if tp + fn == 0:
        recall = 0
    else:
        recall = tp / (tp + fn)

    if precision + recall == 0:
        return 0

    beta = 0.5
    return (1 + beta ** 2) * (precision * recall) / (beta ** 2 * precision + recall)


def bootstrap_ci(scores: List[float], n_bootstrap: int = 1000, ci: float = 0.95) -> Tuple[float, float, float]:
    """Calculate bootstrap confidence interval."""
    boot_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(scores, size=len(scores), replace=True)
        boot_means.append(np.mean(sample))

    lower = np.percentile(boot_means, (1 - ci) / 2 * 100)
    upper = np.percentile(boot_means, (1 + ci) / 2 * 100)
    return np.mean(scores), lower, upper


# ============================================================================
# TEST FUNCTIONS
# ============================================================================

class NahawiEvaluator:
    """Main evaluation class."""

    def __init__(self, model_path: str, tokenizer_path: str, device: str = 'cuda'):
        self.device = device
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(tokenizer_path)

        self.model = NahawiGEC(CONFIG).to(device)
        state = torch.load(model_path, map_location=device)
        self.model.load_state_dict(state.get('model_state_dict', state))
        self.model.eval()

        self.rules = RulePostProcessor()
        self.results = {}

    def correct(self, text: str) -> str:
        """Correct text using neural model + rules."""
        src_ids = torch.tensor([[2] + self.sp.EncodeAsIds(text)[:254] + [3]], device=self.device)
        output = self.model.generate(src_ids)
        ids = output[0].tolist()
        if 3 in ids:
            ids = ids[:ids.index(3)]
        if ids and ids[0] == 2:
            ids = ids[1:]
        return self.rules.process(self.sp.DecodeIds(ids))

    def correct_neural_only(self, text: str) -> str:
        """Correct text using neural model only (for ablation)."""
        src_ids = torch.tensor([[2] + self.sp.EncodeAsIds(text)[:254] + [3]], device=self.device)
        output = self.model.generate(src_ids)
        ids = output[0].tolist()
        if 3 in ids:
            ids = ids[:ids.index(3)]
        if ids and ids[0] == 2:
            ids = ids[1:]
        return self.sp.DecodeIds(ids)

    def correct_rules_only(self, text: str) -> str:
        """Apply rules only (for ablation)."""
        return self.rules.process(text)

    def multi_pass_correct(self, text: str, max_passes: int = 5) -> Tuple[str, int]:
        """Run correction until convergence."""
        current = text
        for i in range(max_passes):
            result = self.correct(current)
            if result == current:
                return result, i + 1
            current = result
        return current, max_passes

    # ========================================================================
    # TEST 1: FASIH BENCHMARK
    # ========================================================================

    def test_fasih_benchmark(self, fasih_path: str) -> Dict:
        """Run full FASIH benchmark evaluation."""
        print("\n" + "=" * 70)
        print("TEST 1: FASIH v4.1 BENCHMARK")
        print("=" * 70)

        with open(fasih_path) as f:
            data = json.load(f)

        # Group by category
        by_category = defaultdict(list)
        for sample in data:
            by_category[sample['error_type']].append(sample)

        results = {}
        category_scores = []

        for category, samples in sorted(by_category.items()):
            tp, fp, fn = 0, 0, 0

            for sample in samples:
                source = sample['source']
                target = sample['target']
                prediction = self.correct(source)

                # Simple token-level comparison
                pred_tokens = set(prediction.split())
                tgt_tokens = set(target.split())
                src_tokens = set(source.split())

                # Corrections made
                pred_corrections = pred_tokens - src_tokens
                gold_corrections = tgt_tokens - src_tokens

                tp += len(pred_corrections & gold_corrections)
                fp += len(pred_corrections - gold_corrections)
                fn += len(gold_corrections - pred_corrections)

            f05 = calculate_f05(tp, fp, fn) * 100
            results[category] = {
                'f05': f05,
                'tp': tp, 'fp': fp, 'fn': fn,
                'samples': len(samples),
                'status': 'PASS' if f05 >= 90 else 'WORK' if f05 >= 80 else 'FAIL'
            }
            category_scores.append(f05)

            status = results[category]['status']
            print(f"  {category:30s}: F0.5={f05:5.1f}% [{status}]")

        # Overall
        overall_f05 = np.mean(category_scores)
        pass_count = sum(1 for r in results.values() if r['status'] == 'PASS')
        total_categories = len(results)

        print("-" * 70)
        print(f"  OVERALL F0.5: {overall_f05:.1f}%")
        print(f"  PASS categories: {pass_count}/{total_categories}")

        # Bootstrap CI
        mean, lower, upper = bootstrap_ci(category_scores)
        print(f"  95% CI: [{lower:.1f}%, {upper:.1f}%]")

        self.results['fasih'] = {
            'overall_f05': overall_f05,
            'ci_lower': lower,
            'ci_upper': upper,
            'pass_count': pass_count,
            'total_categories': total_categories,
            'by_category': results
        }

        return self.results['fasih']

    # ========================================================================
    # TEST 2: FALSE POSITIVE RATE
    # ========================================================================

    def test_false_positive_rate(self, clean_sentences: List[str]) -> Dict:
        """Test false positive rate on clean text."""
        print("\n" + "=" * 70)
        print("TEST 2: FALSE POSITIVE RATE (Clean Text Preservation)")
        print("=" * 70)

        changed = 0
        unchanged = 0
        changes = []

        for sent in clean_sentences:
            output = self.correct(sent)
            if output.strip() != sent.strip():
                changed += 1
                changes.append({'input': sent, 'output': output})
            else:
                unchanged += 1

        total = len(clean_sentences)
        fp_rate = changed / total * 100
        preservation_rate = unchanged / total * 100

        # Bootstrap CI for FP rate
        fp_binary = [1 if c else 0 for c in [self.correct(s).strip() != s.strip() for s in clean_sentences[:100]]]
        mean, lower, upper = bootstrap_ci(fp_binary)

        print(f"\n  Total sentences: {total}")
        print(f"  Unchanged (correct): {unchanged} ({preservation_rate:.1f}%)")
        print(f"  Changed (false positives): {changed} ({fp_rate:.1f}%)")
        print(f"  95% CI for FP rate: [{lower*100:.1f}%, {upper*100:.1f}%]")

        status = 'PASS' if fp_rate < 3 else 'ACCEPTABLE' if fp_rate < 5 else 'NEEDS WORK' if fp_rate < 10 else 'FAIL'
        print(f"\n  STATUS: {status}")

        if changes[:3]:
            print("\n  Sample false positives:")
            for c in changes[:3]:
                print(f"    IN:  {c['input'][:60]}...")
                print(f"    OUT: {c['output'][:60]}...")

        self.results['false_positive'] = {
            'fp_rate': fp_rate,
            'preservation_rate': preservation_rate,
            'total': total,
            'changed': changed,
            'status': status,
            'changes': changes[:20]  # Keep first 20 for analysis
        }

        return self.results['false_positive']

    # ========================================================================
    # TEST 3: ERROR CORRECTION RATE
    # ========================================================================

    def test_error_correction_rate(self, clean_sentences: List[str]) -> Dict:
        """Test error correction rate per category."""
        print("\n" + "=" * 70)
        print("TEST 3: ERROR CORRECTION RATE (Injected Errors)")
        print("=" * 70)

        error_types = ['hamza', 'alif_maqsura', 'taa_marbuta', 'definiteness']
        results = {}

        for error_type in error_types:
            correct_count = 0
            total = 0
            samples_tested = []

            for sent in clean_sentences[:300]:
                corrupted, was_injected = ErrorInjector.inject(sent, error_type)
                if was_injected:
                    total += 1
                    output = self.correct(corrupted)
                    is_correct = output.strip() == sent.strip()
                    if is_correct:
                        correct_count += 1
                    samples_tested.append({
                        'clean': sent,
                        'corrupted': corrupted,
                        'output': output,
                        'correct': is_correct
                    })

            if total > 0:
                rate = correct_count / total * 100
                status = 'PASS' if rate >= 90 else 'WORK' if rate >= 80 else 'FAIL'
                results[error_type] = {
                    'rate': rate,
                    'correct': correct_count,
                    'total': total,
                    'status': status
                }
                print(f"  {error_type:20s}: {rate:5.1f}% ({correct_count}/{total}) [{status}]")
            else:
                print(f"  {error_type:20s}: No samples")

        self.results['error_correction'] = results
        return results

    # ========================================================================
    # TEST 4: ROBUSTNESS (EDGE CASES)
    # ========================================================================

    def test_edge_cases(self) -> Dict:
        """Test edge cases: short, long, mixed content."""
        print("\n" + "=" * 70)
        print("TEST 4: EDGE CASES (Robustness)")
        print("=" * 70)

        results = {}

        # Short sentences
        short_tests = [
            ("ذهب إلى المدرسة.", True),  # Should preserve
            ("ذهب الي المدرسه.", False),  # Should correct
            ("كتاب جميل.", True),
            ("هذا صحيح.", True),
            ("نعم، أوافق.", True),
        ]

        short_pass = 0
        for text, should_preserve in short_tests:
            output = self.correct(text)
            if should_preserve:
                passed = output.strip() == text.strip()
            else:
                passed = output.strip() != text.strip()
            if passed:
                short_pass += 1

        results['short'] = {'passed': short_pass, 'total': len(short_tests)}
        print(f"\n  Short sentences: {short_pass}/{len(short_tests)} passed")

        # Long sentences
        long_tests = [
            "تعتبر اللغة العربية من أقدم اللغات السامية وأكثرها انتشاراً في العالم، حيث يتحدث بها أكثر من أربعمائة مليون نسمة، وهي اللغة الرسمية في اثنتين وعشرين دولة عربية.",
            "يمتد تاريخ الحضارة العربية الإسلامية على مدى قرون طويلة، شهدت فيها ازدهاراً علمياً وثقافياً كبيراً، أسهم في تطور العلوم والفنون والآداب في مختلف أنحاء العالم.",
        ]

        long_pass = 0
        for text in long_tests:
            output = self.correct(text)
            if output.strip() == text.strip():
                long_pass += 1

        results['long'] = {'passed': long_pass, 'total': len(long_tests)}
        print(f"  Long sentences: {long_pass}/{len(long_tests)} preserved")

        # Mixed content (numbers, names)
        mixed_tests = [
            "ولد الكاتب محمد في عام 1985 في مدينة القاهرة.",
            "بلغت المبيعات 500 مليون دولار في الربع الأول.",
            "التقى الرئيس أوباما بنظيره الفرنسي ماكرون.",
        ]

        mixed_pass = 0
        for text in mixed_tests:
            output = self.correct(text)
            if output.strip() == text.strip():
                mixed_pass += 1

        results['mixed'] = {'passed': mixed_pass, 'total': len(mixed_tests)}
        print(f"  Mixed content: {mixed_pass}/{len(mixed_tests)} preserved")

        self.results['edge_cases'] = results
        return results

    # ========================================================================
    # TEST 5: MULTI-PASS CONVERGENCE
    # ========================================================================

    def test_multi_pass(self) -> Dict:
        """Test multi-pass correction for heavily corrupted text."""
        print("\n" + "=" * 70)
        print("TEST 5: MULTI-PASS CONVERGENCE")
        print("=" * 70)

        test_cases = [
            {'corrupted': 'ذهب الي المدرسه الكبيره', 'correct': 'ذهب إلى المدرسة الكبيرة', 'errors': 3},
            {'corrupted': 'هذه الجامعه الكبيره في المدينه', 'correct': 'هذه الجامعة الكبيرة في المدينة', 'errors': 3},
            {'corrupted': 'يجب ان نذهب الي هناك حتي نري الحقيقه', 'correct': 'يجب أن نذهب إلى هناك حتى نرى الحقيقة', 'errors': 5},
        ]

        results = {'fully_corrected': 0, 'partial': 0, 'passes_needed': []}

        for tc in test_cases:
            final, passes = self.multi_pass_correct(tc['corrupted'])
            is_correct = final.strip() == tc['correct'].strip()

            if is_correct:
                results['fully_corrected'] += 1
                results['passes_needed'].append(passes)
                print(f"  ✓ {tc['corrupted'][:40]}... → {passes} passes")
            else:
                results['partial'] += 1
                print(f"  ○ {tc['corrupted'][:40]}... → partial after {passes} passes")

        results['total'] = len(test_cases)
        results['success_rate'] = results['fully_corrected'] / results['total'] * 100

        if results['passes_needed']:
            results['avg_passes'] = np.mean(results['passes_needed'])
        else:
            results['avg_passes'] = 0

        print(f"\n  Success rate: {results['success_rate']:.0f}%")
        if results['passes_needed']:
            print(f"  Average passes: {results['avg_passes']:.1f}")

        self.results['multi_pass'] = results
        return results

    # ========================================================================
    # TEST 6: ABLATION STUDIES
    # ========================================================================

    def test_ablation(self, test_samples: List[Dict]) -> Dict:
        """Run ablation studies to measure component contribution."""
        print("\n" + "=" * 70)
        print("TEST 6: ABLATION STUDIES")
        print("=" * 70)

        results = {}

        # Full system
        full_correct = sum(1 for s in test_samples if self.correct(s['source']).strip() == s['target'].strip())
        results['full_system'] = full_correct / len(test_samples) * 100

        # Neural only
        neural_correct = sum(1 for s in test_samples if self.correct_neural_only(s['source']).strip() == s['target'].strip())
        results['neural_only'] = neural_correct / len(test_samples) * 100

        # Rules only
        rules_correct = sum(1 for s in test_samples if self.correct_rules_only(s['source']).strip() == s['target'].strip())
        results['rules_only'] = rules_correct / len(test_samples) * 100

        print(f"\n  Full system:  {results['full_system']:.1f}%")
        print(f"  Neural only:  {results['neural_only']:.1f}% (delta: {results['full_system'] - results['neural_only']:+.1f}%)")
        print(f"  Rules only:   {results['rules_only']:.1f}%")

        # Calculate rule contribution
        results['rule_contribution'] = results['full_system'] - results['neural_only']

        self.results['ablation'] = results
        return results

    # ========================================================================
    # GENERATE REPORT
    # ========================================================================

    def generate_report(self, output_path: str = 'evaluation_report.json'):
        """Generate comprehensive evaluation report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_config': CONFIG,
            'random_seed': SEED,
            'results': self.results,
            'summary': {
                'production_ready': self._is_production_ready(),
                'key_metrics': self._get_key_metrics()
            }
        }

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"\n\nReport saved to: {output_path}")
        return report

    def _is_production_ready(self) -> bool:
        """Check if system meets production criteria."""
        checks = []

        if 'fasih' in self.results:
            checks.append(self.results['fasih']['overall_f05'] >= 90)
            checks.append(self.results['fasih']['pass_count'] >= 10)

        if 'false_positive' in self.results:
            checks.append(self.results['false_positive']['fp_rate'] < 5)

        return all(checks) if checks else False

    def _get_key_metrics(self) -> Dict:
        """Extract key metrics for summary."""
        metrics = {}

        if 'fasih' in self.results:
            metrics['fasih_f05'] = f"{self.results['fasih']['overall_f05']:.1f}%"
            metrics['fasih_ci'] = f"[{self.results['fasih']['ci_lower']:.1f}%, {self.results['fasih']['ci_upper']:.1f}%]"

        if 'false_positive' in self.results:
            metrics['fp_rate'] = f"{self.results['false_positive']['fp_rate']:.1f}%"

        if 'multi_pass' in self.results:
            metrics['multi_pass_success'] = f"{self.results['multi_pass']['success_rate']:.0f}%"

        return metrics


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Nahawi Scientific Evaluation Suite')
    parser.add_argument('--all', action='store_true', help='Run all tests')
    parser.add_argument('--benchmark', action='store_true', help='Run FASIH benchmark')
    parser.add_argument('--fp-test', action='store_true', help='Run false positive test')
    parser.add_argument('--cross-domain', action='store_true', help='Run cross-domain test')
    parser.add_argument('--robustness', action='store_true', help='Run robustness tests')
    parser.add_argument('--ablation', action='store_true', help='Run ablation studies')
    parser.add_argument('--model-path', default='fasih_v5_model/best_model.pt')
    parser.add_argument('--tokenizer-path', default='nahawi_spm.model')
    parser.add_argument('--fasih-path', default='data/fasih_v4.1/fasih_test.json')
    parser.add_argument('--output', default='evaluation_report.json')

    args = parser.parse_args()

    # Default to all if no specific test selected
    if not any([args.all, args.benchmark, args.fp_test, args.cross_domain, args.robustness, args.ablation]):
        args.all = True

    print("=" * 70)
    print("NAHAWI SCIENTIFIC EVALUATION SUITE")
    print("=" * 70)
    print(f"\nModel: {args.model_path}")
    print(f"Tokenizer: {args.tokenizer_path}")
    print(f"Random seed: {SEED}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Initialize evaluator
    print("\nLoading model...")
    evaluator = NahawiEvaluator(args.model_path, args.tokenizer_path, device)
    print("Model loaded!")

    # Load test data
    with open(args.fasih_path) as f:
        fasih_data = json.load(f)

    clean_sentences = list(set([s['target'] for s in fasih_data]))[:500]
    print(f"Loaded {len(clean_sentences)} clean sentences")

    # Run tests
    if args.all or args.benchmark:
        evaluator.test_fasih_benchmark(args.fasih_path)

    if args.all or args.fp_test:
        evaluator.test_false_positive_rate(clean_sentences)

    if args.all or args.robustness:
        evaluator.test_error_correction_rate(clean_sentences)
        evaluator.test_edge_cases()
        evaluator.test_multi_pass()

    if args.all or args.ablation:
        test_samples = [{'source': s['source'], 'target': s['target']} for s in fasih_data[:200]]
        evaluator.test_ablation(test_samples)

    # Generate report
    report = evaluator.generate_report(args.output)

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    for key, value in report['summary']['key_metrics'].items():
        print(f"  {key}: {value}")

    print(f"\n  PRODUCTION READY: {'YES ✓' if report['summary']['production_ready'] else 'NO ✗'}")


if __name__ == '__main__':
    main()
