#!/usr/bin/env python3
"""
Generate targeted training data for the 3 fixable weak categories:
1. Verb conjugation: وا→ون (20K pairs)
2. Hamza swap: ئ↔ؤ seat changes (20K pairs)
3. Case endings: ين→ون for sound masculine plurals (20K pairs)

Uses MSA corpus to find real examples.
"""

import json
import random
import re
from pathlib import Path

MSA_CORPUS = "/home/ubuntu/nahawi/corpus/combined/msa_corpus_full.txt"
OUTPUT_DIR = "/home/ubuntu/nahawi/data"

# Common verb roots that take وا/ون endings
VERB_ROOTS = [
    'يتعلم', 'يعمل', 'يذهب', 'يكتب', 'يقرأ', 'يفهم', 'يسمع', 'يرى',
    'يعرف', 'يأخذ', 'يجد', 'يقول', 'يريد', 'يستطيع', 'يحب', 'يعتقد',
    'يظن', 'يشعر', 'يفكر', 'يتكلم', 'يسافر', 'يدرس', 'يعالج', 'يساعد',
    'يحاول', 'يستمر', 'يبدأ', 'ينتهي', 'يصل', 'يخرج', 'يدخل', 'يرجع',
    'يتحدث', 'يشارك', 'يستخدم', 'يحتاج', 'يقدم', 'يتوقع', 'يتمنى', 'يتذكر',
]

# Common sound masculine plural nouns
MASCULINE_PLURALS = [
    ('المحامي', 'المحامين', 'المحامون'),
    ('المدير', 'المديرين', 'المديرون'),
    ('الباحث', 'الباحثين', 'الباحثون'),
    ('المعلم', 'المعلمين', 'المعلمون'),
    ('المهندس', 'المهندسين', 'المهندسون'),
    ('الموظف', 'الموظفين', 'الموظفون'),
    ('العامل', 'العاملين', 'العاملون'),
    ('المسلم', 'المسلمين', 'المسلمون'),
    ('المؤمن', 'المؤمنين', 'المؤمنون'),
    ('الفائز', 'الفائزين', 'الفائزون'),
    ('اللاعب', 'اللاعبين', 'اللاعبون'),
    ('المشارك', 'المشاركين', 'المشاركون'),
    ('الحاضر', 'الحاضرين', 'الحاضرون'),
    ('المتقدم', 'المتقدمين', 'المتقدمون'),
    ('المنتج', 'المنتجين', 'المنتجون'),
    ('المستثمر', 'المستثمرين', 'المستثمرون'),
    ('المسافر', 'المسافرين', 'المسافرون'),
    ('المتخصص', 'المتخصصين', 'المتخصصون'),
    ('الناخب', 'الناخبين', 'الناخبون'),
    ('المراقب', 'المراقبين', 'المراقبون'),
]

# Hamza swap patterns (ئ on yaa seat vs ؤ on waw seat)
HAMZA_SWAP_WORDS = [
    ('مسئول', 'مسؤول'),
    ('مسئولية', 'مسؤولية'),
    ('رئيس', 'رؤساء'),  # Different pattern but hamza related
    ('تساؤل', 'تساءل'),
    ('شئون', 'شؤون'),
    ('فئة', 'فئة'),  # Keep as is
    ('بيئة', 'بيئة'),  # Keep as is
    ('هيئة', 'هيئة'),  # Keep as is
]


def generate_verb_conjugation_pairs(corpus_lines, target_count=20000):
    """Generate وا→ون verb conjugation pairs."""
    pairs = []

    # Pattern: verb + وا at end (incorrect) should be verb + ون (correct)
    # Context: when preceded by لم، لن، أن، حتى، كي، ل (subjunctive/jussive triggers)

    subjunctive_triggers = ['أن', 'لن', 'كي', 'حتى', 'ل']

    for line in corpus_lines:
        words = line.split()
        for i, word in enumerate(words):
            # Find verbs ending in ون (correct form)
            if word.endswith('ون') and len(word) > 3:
                # Check if preceded by subjunctive trigger
                if i > 0 and any(words[i-1].endswith(t) or words[i-1] == t for t in subjunctive_triggers):
                    # Create error: ون → وا
                    error_word = word[:-2] + 'وا'

                    # Build context
                    context_start = max(0, i - 3)
                    context_end = min(len(words), i + 4)

                    src_words = words[context_start:context_end].copy()
                    tgt_words = words[context_start:context_end].copy()

                    # Replace in source with error
                    src_idx = i - context_start
                    src_words[src_idx] = error_word

                    src = ' '.join(src_words)
                    tgt = ' '.join(tgt_words)

                    if src != tgt and len(src) > 10:
                        pairs.append({'source': src, 'target': tgt})

                        if len(pairs) >= target_count:
                            return pairs

    # If not enough from corpus, generate synthetic
    print(f"  Found {len(pairs)} from corpus, generating synthetic...")

    contexts = [
        'يجب أن {} في الوقت المحدد',
        'لن {} إلا بعد الموافقة',
        'حتى {} على النتائج',
        'كي {} بشكل صحيح',
        'عليهم أن {} قبل الموعد',
        'من المهم أن {} جيداً',
        'يريدون أن {} في المشروع',
        'لابد أن {} مع الفريق',
    ]

    while len(pairs) < target_count:
        root = random.choice(VERB_ROOTS)
        context = random.choice(contexts)

        correct = root + 'ون'
        error = root + 'وا'

        tgt = context.format(correct)
        src = context.format(error)

        pairs.append({'source': src, 'target': tgt})

    return pairs[:target_count]


def generate_hamza_swap_pairs(corpus_lines, target_count=20000):
    """Generate ئ↔ؤ hamza seat swap pairs."""
    pairs = []

    # Main pattern: مسئول (Egyptian/Levantine) → مسؤول (MSA standard)
    swap_map = {
        'مسئول': 'مسؤول',
        'مسئولية': 'مسؤولية',
        'مسئولين': 'مسؤولين',
        'مسئولون': 'مسؤولون',
        'مسئوليات': 'مسؤوليات',
        'شئون': 'شؤون',
        'تسائل': 'تساؤل',
        'تسائلات': 'تساؤلات',
    }

    for line in corpus_lines:
        # Check for correct forms and create error pairs
        for correct, _ in [('مسؤول', 'مسئول'), ('مسؤولية', 'مسئولية'),
                           ('مسؤولين', 'مسئولين'), ('مسؤولون', 'مسئولون'),
                           ('شؤون', 'شئون')]:
            if correct in line:
                error = correct.replace('ؤ', 'ئ')
                src = line.replace(correct, error)
                tgt = line

                if src != tgt and len(src) > 10 and len(src) < 200:
                    pairs.append({'source': src, 'target': tgt})

                    if len(pairs) >= target_count:
                        return pairs

    # Generate synthetic if needed
    print(f"  Found {len(pairs)} from corpus, generating synthetic...")

    contexts = [
        'تحدث {} الشركة عن الخطة الجديدة',
        'يعتبر {} عن هذا القرار',
        'تقع {} على عاتق الإدارة',
        'أكد {} الحكومة أهمية المشروع',
        'تتحمل الدولة {} كبيرة',
        'من {} الجميع المشاركة',
        'يجب أن يكون {} عن أفعاله',
        'تعد هذه {} الرئيسية للفريق',
    ]

    error_correct_pairs = [
        ('مسئول', 'مسؤول'),
        ('مسئولية', 'مسؤولية'),
        ('شئون', 'شؤون'),
    ]

    while len(pairs) < target_count:
        error, correct = random.choice(error_correct_pairs)
        context = random.choice(contexts)

        src = context.format(error)
        tgt = context.format(correct)

        pairs.append({'source': src, 'target': tgt})

    return pairs[:target_count]


def generate_case_ending_pairs(corpus_lines, target_count=20000):
    """Generate ين→ون case ending pairs for sound masculine plurals."""
    pairs = []

    # Pattern: nominative case requires ون, accusative/genitive requires ين
    # Error: using ين when ون is needed (subject position)

    nominative_contexts = [
        'جاء {}',
        'حضر {}',
        'قال {}',
        'أعلن {}',
        'صرح {}',
        'أكد {}',
        'يعمل {}',
        'يسعى {}',
        '{} يعملون',
        '{} حاضرون',
        '{} مسؤولون',
    ]

    for line in corpus_lines:
        words = line.split()
        for i, word in enumerate(words):
            # Find words ending in ون (correct nominative)
            if word.endswith('ون') and word.startswith('ال') and len(word) > 5:
                # Check if in subject position (after verb or at start)
                if i == 0 or (i > 0 and len(words[i-1]) > 2):
                    # Create error: ون → ين
                    error_word = word[:-2] + 'ين'

                    context_start = max(0, i - 2)
                    context_end = min(len(words), i + 3)

                    src_words = words[context_start:context_end].copy()
                    tgt_words = words[context_start:context_end].copy()

                    src_idx = i - context_start
                    src_words[src_idx] = error_word

                    src = ' '.join(src_words)
                    tgt = ' '.join(tgt_words)

                    if src != tgt and len(src) > 10:
                        pairs.append({'source': src, 'target': tgt})

                        if len(pairs) >= target_count:
                            return pairs

    # Generate synthetic
    print(f"  Found {len(pairs)} from corpus, generating synthetic...")

    while len(pairs) < target_count:
        _, yin_form, wun_form = random.choice(MASCULINE_PLURALS)
        context = random.choice(nominative_contexts)

        # In nominative (subject) position, ون is correct
        src = context.format(yin_form)  # Error: ين
        tgt = context.format(wun_form)  # Correct: ون

        pairs.append({'source': src, 'target': tgt})

    return pairs[:target_count]


def main():
    print("=" * 70)
    print("GENERATING TARGETED TRAINING DATA")
    print("=" * 70)

    # Load corpus
    print("\nLoading MSA corpus...")
    corpus_lines = []
    with open(MSA_CORPUS, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if len(line) > 20 and len(line) < 300:
                corpus_lines.append(line)
                if len(corpus_lines) >= 2000000:  # Limit for memory
                    break
    print(f"  Loaded {len(corpus_lines):,} lines")

    # Shuffle for variety
    random.shuffle(corpus_lines)

    # Generate each category
    print("\n1. Generating verb conjugation pairs (وا→ون)...")
    verb_pairs = generate_verb_conjugation_pairs(corpus_lines, 20000)
    print(f"   Generated {len(verb_pairs):,} pairs")

    print("\n2. Generating hamza swap pairs (ئ↔ؤ)...")
    hamza_pairs = generate_hamza_swap_pairs(corpus_lines, 20000)
    print(f"   Generated {len(hamza_pairs):,} pairs")

    print("\n3. Generating case ending pairs (ين→ون)...")
    case_pairs = generate_case_ending_pairs(corpus_lines, 20000)
    print(f"   Generated {len(case_pairs):,} pairs")

    # Save individual files
    output_dir = Path(OUTPUT_DIR)

    with open(output_dir / 'verb_conjugation_train.json', 'w', encoding='utf-8') as f:
        json.dump(verb_pairs, f, ensure_ascii=False, indent=2)
    print(f"\nSaved verb_conjugation_train.json ({len(verb_pairs):,} pairs)")

    with open(output_dir / 'hamza_swap_train.json', 'w', encoding='utf-8') as f:
        json.dump(hamza_pairs, f, ensure_ascii=False, indent=2)
    print(f"Saved hamza_swap_train.json ({len(hamza_pairs):,} pairs)")

    with open(output_dir / 'case_ending_train.json', 'w', encoding='utf-8') as f:
        json.dump(case_pairs, f, ensure_ascii=False, indent=2)
    print(f"Saved case_ending_train.json ({len(case_pairs):,} pairs)")

    # Merge all into one file
    all_pairs = verb_pairs + hamza_pairs + case_pairs
    random.shuffle(all_pairs)

    with open(output_dir / 'targeted_fixes_train.json', 'w', encoding='utf-8') as f:
        json.dump(all_pairs, f, ensure_ascii=False, indent=2)
    print(f"\nSaved targeted_fixes_train.json ({len(all_pairs):,} total pairs)")

    # Show samples
    print("\n" + "=" * 70)
    print("SAMPLES")
    print("=" * 70)

    print("\nVerb conjugation:")
    for p in verb_pairs[:3]:
        print(f"  src: {p['source']}")
        print(f"  tgt: {p['target']}\n")

    print("Hamza swap:")
    for p in hamza_pairs[:3]:
        print(f"  src: {p['source']}")
        print(f"  tgt: {p['target']}\n")

    print("Case endings:")
    for p in case_pairs[:3]:
        print(f"  src: {p['source']}")
        print(f"  tgt: {p['target']}\n")


if __name__ == "__main__":
    main()
