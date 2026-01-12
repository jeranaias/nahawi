#!/usr/bin/env python3
"""
Extract REAL patterns from MSA corpus for weak categories.
Same playbook as hamza augmentation - real data, not synthetic.

Categories:
1. Verb conjugation: يتعلمون → يتعلموا (formal ن dropped in colloquial)
2. Hamza swap: مسؤول → مسئول (ؤ↔ئ seat variation)
3. Case endings: المحامون → المحامين (nominative ون → accusative ين)
4. Feminine agreement: رئيسية → رئيسي (missing ة)

Target: 20K pairs per category, 80K total.
"""

import json
import random
import re
from pathlib import Path
from collections import defaultdict

MSA_CORPUS = "/home/ubuntu/nahawi/corpus/combined/msa_corpus_full.txt"
OUTPUT_DIR = "/home/ubuntu/nahawi/data"
TARGET_PER_CATEGORY = 20000

# Patterns for extraction
VERB_CONJUGATION_ENDINGS = ['ون', 'ين']  # Present tense masculine plural
HAMZA_WAW_WORDS = ['مسؤول', 'مسؤولية', 'مسؤولين', 'مسؤولون', 'مسؤوليات',
                   'رؤية', 'رؤى', 'رؤساء', 'رؤوس',
                   'مؤتمر', 'مؤتمرات', 'مؤسسة', 'مؤسسات',
                   'شؤون', 'مؤلف', 'مؤلفات', 'مؤشر', 'مؤشرات',
                   'تساؤل', 'تساؤلات', 'مؤهل', 'مؤهلات', 'مؤثر',
                   'فؤاد', 'سؤال', 'أسؤلة', 'مؤمن', 'مؤمنون']

NOMINATIVE_PLURALS = ['المحامون', 'المديرون', 'الباحثون', 'المعلمون', 'المهندسون',
                      'الموظفون', 'العاملون', 'المسلمون', 'المؤمنون', 'الفائزون',
                      'اللاعبون', 'المشاركون', 'الحاضرون', 'المتقدمون', 'المنتجون',
                      'المستثمرون', 'المسافرون', 'المتخصصون', 'الناخبون', 'المراقبون',
                      'المصريون', 'السعوديون', 'العراقيون', 'السوريون', 'اللبنانيون',
                      'الفلسطينيون', 'الأردنيون', 'الكويتيون', 'الإماراتيون', 'القطريون',
                      'المغاربة', 'التونسيون', 'الجزائريون', 'الليبيون', 'السودانيون',
                      'اليمنيون', 'العمانيون', 'البحرينيون', 'المواطنون', 'المسؤولون',
                      'الصحفيون', 'الإعلاميون', 'الأطباء', 'المحللون', 'الخبراء']

FEMININE_ADJECTIVES = ['رئيسية', 'متكاملة', 'محلية', 'دولية', 'عربية',
                       'جديدة', 'قديمة', 'كبيرة', 'صغيرة', 'طويلة',
                       'قصيرة', 'عالية', 'منخفضة', 'سريعة', 'بطيئة',
                       'قوية', 'ضعيفة', 'جميلة', 'قبيحة', 'سعيدة',
                       'حزينة', 'غنية', 'فقيرة', 'صحية', 'مرضية',
                       'تعليمية', 'اقتصادية', 'سياسية', 'اجتماعية', 'ثقافية',
                       'تاريخية', 'جغرافية', 'علمية', 'أدبية', 'فنية',
                       'رياضية', 'عسكرية', 'أمنية', 'قانونية', 'إدارية',
                       'مالية', 'تجارية', 'صناعية', 'زراعية', 'بيئية',
                       'تقنية', 'رقمية', 'إلكترونية', 'افتراضية', 'حقيقية']


def extract_verb_conjugation(lines, target=20000):
    """Extract real verb conjugation patterns (يـون → يـوا)."""
    pairs = []
    seen = set()

    # Pattern: verbs ending in ون preceded by ي (present tense)
    verb_pattern = re.compile(r'\bي\w+ون\b')

    for line in lines:
        if len(pairs) >= target:
            break

        words = line.split()
        for i, word in enumerate(words):
            if verb_pattern.match(word) and word.endswith('ون') and len(word) >= 5:
                # Create corrupted version: ون → وا
                corrupted = word[:-2] + 'وا'

                # Build context (3 words before and after)
                start = max(0, i - 3)
                end = min(len(words), i + 4)
                context = words[start:end]

                # Create pair
                src_context = context.copy()
                tgt_context = context.copy()

                word_idx = i - start
                src_context[word_idx] = corrupted

                src = ' '.join(src_context)
                tgt = ' '.join(tgt_context)

                if src != tgt and len(src) > 15 and (src, tgt) not in seen:
                    seen.add((src, tgt))
                    pairs.append({'source': src, 'target': tgt})

    return pairs


def extract_hamza_swap(lines, target=20000):
    """Extract real hamza swap patterns (ؤ → ئ)."""
    pairs = []
    seen = set()

    for line in lines:
        if len(pairs) >= target:
            break

        for correct_word in HAMZA_WAW_WORDS:
            if correct_word in line:
                # Create corrupted version: ؤ → ئ
                corrupted_word = correct_word.replace('ؤ', 'ئ')

                if corrupted_word != correct_word:
                    src = line.replace(correct_word, corrupted_word)
                    tgt = line

                    # Limit length
                    if len(src) > 200:
                        # Find the word and extract context
                        words = line.split()
                        for i, w in enumerate(words):
                            if correct_word in w:
                                start = max(0, i - 4)
                                end = min(len(words), i + 5)
                                context = words[start:end]
                                tgt = ' '.join(context)
                                src = tgt.replace(correct_word, corrupted_word)
                                break

                    if src != tgt and len(src) > 10 and (src, tgt) not in seen:
                        seen.add((src, tgt))
                        pairs.append({'source': src, 'target': tgt})
                        break  # One pair per line

    return pairs


def extract_case_endings(lines, target=20000):
    """Extract real case ending patterns (ون → ين in nominative context)."""
    pairs = []
    seen = set()

    for line in lines:
        if len(pairs) >= target:
            break

        for nom_word in NOMINATIVE_PLURALS:
            if nom_word in line:
                # Create corrupted version: ون → ين
                corrupted_word = nom_word[:-2] + 'ين'

                # Find context
                words = line.split()
                for i, w in enumerate(words):
                    if w == nom_word or nom_word in w:
                        start = max(0, i - 3)
                        end = min(len(words), i + 4)
                        context = words[start:end]

                        tgt = ' '.join(context)
                        src = tgt.replace(nom_word, corrupted_word)

                        if src != tgt and len(src) > 10 and (src, tgt) not in seen:
                            seen.add((src, tgt))
                            pairs.append({'source': src, 'target': tgt})
                            break
                break  # One pair per line

    return pairs


def extract_feminine_agreement(lines, target=20000):
    """Extract real feminine agreement patterns (ة → missing)."""
    pairs = []
    seen = set()

    for line in lines:
        if len(pairs) >= target:
            break

        for fem_word in FEMININE_ADJECTIVES:
            if fem_word in line:
                # Create corrupted version: remove final ة
                corrupted_word = fem_word[:-1]

                # Make sure it's a standalone word (not part of larger word)
                if f' {fem_word}' in line or line.startswith(fem_word):
                    # Find context
                    words = line.split()
                    for i, w in enumerate(words):
                        if w == fem_word:
                            start = max(0, i - 3)
                            end = min(len(words), i + 4)
                            context = words[start:end]

                            tgt = ' '.join(context)
                            src = tgt.replace(fem_word, corrupted_word)

                            if src != tgt and len(src) > 10 and (src, tgt) not in seen:
                                seen.add((src, tgt))
                                pairs.append({'source': src, 'target': tgt})
                                break
                    break  # One pair per line

    return pairs


def main():
    print("=" * 70)
    print("EXTRACTING REAL PATTERNS FROM MSA CORPUS")
    print("=" * 70)

    # Load corpus
    print(f"\nLoading corpus from {MSA_CORPUS}...")
    lines = []
    with open(MSA_CORPUS, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if 20 < len(line) < 500:  # Reasonable length
                lines.append(line)

    print(f"Loaded {len(lines):,} lines")
    random.shuffle(lines)

    # Extract each category
    print(f"\n1. Extracting verb conjugation (يـون → يـوا)...")
    verb_pairs = extract_verb_conjugation(lines, TARGET_PER_CATEGORY)
    print(f"   Found {len(verb_pairs):,} pairs")

    print(f"\n2. Extracting hamza swap (ؤ → ئ)...")
    hamza_pairs = extract_hamza_swap(lines, TARGET_PER_CATEGORY)
    print(f"   Found {len(hamza_pairs):,} pairs")

    print(f"\n3. Extracting case endings (ون → ين)...")
    case_pairs = extract_case_endings(lines, TARGET_PER_CATEGORY)
    print(f"   Found {len(case_pairs):,} pairs")

    print(f"\n4. Extracting feminine agreement (ة → missing)...")
    fem_pairs = extract_feminine_agreement(lines, TARGET_PER_CATEGORY)
    print(f"   Found {len(fem_pairs):,} pairs")

    # Save individual files
    output_dir = Path(OUTPUT_DIR)

    with open(output_dir / 'real_verb_conj.json', 'w', encoding='utf-8') as f:
        json.dump(verb_pairs, f, ensure_ascii=False, indent=2)

    with open(output_dir / 'real_hamza_swap.json', 'w', encoding='utf-8') as f:
        json.dump(hamza_pairs, f, ensure_ascii=False, indent=2)

    with open(output_dir / 'real_case_endings.json', 'w', encoding='utf-8') as f:
        json.dump(case_pairs, f, ensure_ascii=False, indent=2)

    with open(output_dir / 'real_feminine.json', 'w', encoding='utf-8') as f:
        json.dump(fem_pairs, f, ensure_ascii=False, indent=2)

    # Combine all
    all_pairs = verb_pairs + hamza_pairs + case_pairs + fem_pairs
    random.shuffle(all_pairs)

    with open(output_dir / 'real_patterns_all.json', 'w', encoding='utf-8') as f:
        json.dump(all_pairs, f, ensure_ascii=False, indent=2)

    print(f"\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Verb conjugation:    {len(verb_pairs):,} pairs")
    print(f"Hamza swap:          {len(hamza_pairs):,} pairs")
    print(f"Case endings:        {len(case_pairs):,} pairs")
    print(f"Feminine agreement:  {len(fem_pairs):,} pairs")
    print(f"TOTAL:               {len(all_pairs):,} pairs")

    # Show samples
    print(f"\n" + "=" * 70)
    print("SAMPLES")
    print("=" * 70)

    if verb_pairs:
        print("\nVerb conjugation:")
        for p in verb_pairs[:2]:
            print(f"  src: {p['source']}")
            print(f"  tgt: {p['target']}\n")

    if hamza_pairs:
        print("Hamza swap:")
        for p in hamza_pairs[:2]:
            print(f"  src: {p['source']}")
            print(f"  tgt: {p['target']}\n")

    if case_pairs:
        print("Case endings:")
        for p in case_pairs[:2]:
            print(f"  src: {p['source']}")
            print(f"  tgt: {p['target']}\n")

    if fem_pairs:
        print("Feminine agreement:")
        for p in fem_pairs[:2]:
            print(f"  src: {p['source']}")
            print(f"  tgt: {p['target']}\n")


if __name__ == "__main__":
    main()
