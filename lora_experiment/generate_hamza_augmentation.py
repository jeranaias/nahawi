#!/usr/bin/env python3
"""
Generate hamza augmentation data from MSA corpus.

Strategy:
1. Find sentences with proper hamza usage (إعلان، أولوية، إلى، آخر)
2. Corrupt them by stripping hamza (اعلان، اولوية، الي، اخر)
3. Create (corrupted, correct) pairs for LoRA training
"""

import os
import re
import json
import random
from collections import defaultdict
from pathlib import Path

# Hamza patterns to find and corrupt
HAMZA_PATTERNS = {
    # إ on alif with kasra -> bare alif
    'إ': 'ا',
    # أ on alif with fatha -> bare alif
    'أ': 'ا',
    # آ alif madda -> bare alif
    'آ': 'ا',
    # ؤ hamza on waw -> bare waw
    'ؤ': 'و',
    # ئ hamza on yaa -> bare yaa
    'ئ': 'ي',
}

# Common words that should have hamza (high-value targets)
HIGH_VALUE_WORDS = {
    'إلى': 'الي',
    'إن': 'ان',
    'إذا': 'اذا',
    'إعلان': 'اعلان',
    'إعلام': 'اعلام',
    'إنسان': 'انسان',
    'إيران': 'ايران',
    'إسرائيل': 'اسرائيل',
    'إسلام': 'اسلام',
    'إضافة': 'اضافة',
    'إجراء': 'اجراء',
    'إدارة': 'ادارة',
    'إطار': 'اطار',
    'أن': 'ان',
    'أو': 'او',
    'أي': 'اي',
    'أكثر': 'اكثر',
    'أقل': 'اقل',
    'أول': 'اول',
    'أولى': 'اولى',
    'أولوية': 'اولوية',
    'أهم': 'اهم',
    'أهمية': 'اهمية',
    'أخرى': 'اخرى',
    'أخير': 'اخير',
    'أساس': 'اساس',
    'أساسي': 'اساسي',
    'أمام': 'امام',
    'أمر': 'امر',
    'أمن': 'امن',
    'أمريكا': 'امريكا',
    'أمريكي': 'امريكي',
    'آخر': 'اخر',
    'آلاف': 'الاف',
    'آن': 'ان',
    'مسؤول': 'مسوول',
    'مسؤولية': 'مسوولية',
    'رؤية': 'رويه',
    'رئيس': 'رئيس',  # This one has hamza on yaa
    'شؤون': 'شوون',
    'فؤاد': 'فواد',
    'سؤال': 'سوال',
    'تساؤل': 'تساول',
}

# Reverse mapping for detection
CORRUPT_TO_CORRECT = {v: k for k, v in HIGH_VALUE_WORDS.items()}


def has_hamza(text):
    """Check if text contains any hamza characters."""
    return bool(re.search(r'[إأآؤئ]', text))


def corrupt_hamza(text):
    """Strip hamza from text to create corrupted version."""
    result = text
    for correct, corrupt in HAMZA_PATTERNS.items():
        result = result.replace(correct, corrupt)
    return result


def corrupt_sentence_selective(sentence, corruption_rate=0.7):
    """Corrupt hamza in sentence with some probability per word."""
    words = sentence.split()
    corrupted_words = []
    changes_made = 0

    for word in words:
        if has_hamza(word) and random.random() < corruption_rate:
            corrupted_words.append(corrupt_hamza(word))
            changes_made += 1
        else:
            corrupted_words.append(word)

    return ' '.join(corrupted_words), changes_made


def process_msa_corpus(corpus_dir, output_file, target_pairs=100000):
    """Process MSA corpus and generate hamza augmentation pairs."""

    pairs = []
    high_value_pairs = []
    stats = defaultdict(int)

    # Find all text files
    corpus_path = Path(corpus_dir)
    text_files = list(corpus_path.glob('**/*.txt'))

    if not text_files:
        # Try looking for jsonl files
        text_files = list(corpus_path.glob('**/*.jsonl'))

    print(f"Found {len(text_files)} corpus files", flush=True)

    for file_path in text_files:
        print(f"Processing {file_path.name}...", flush=True)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    if line_num % 100000 == 0 and line_num > 0:
                        print(f"  {line_num:,} lines, {len(pairs):,} pairs", flush=True)

                    # Handle both plain text and jsonl
                    if file_path.suffix == '.jsonl':
                        try:
                            data = json.loads(line)
                            text = data.get('text', data.get('content', ''))
                        except:
                            continue
                    else:
                        text = line.strip()

                    if not text or len(text) < 10:
                        continue

                    # Split into sentences
                    sentences = re.split(r'[.!؟\n]', text)

                    for sent in sentences:
                        sent = sent.strip()
                        if len(sent) < 15 or len(sent) > 200:
                            continue

                        if not has_hamza(sent):
                            continue

                        # Check for high-value words
                        has_high_value = any(w in sent for w in HIGH_VALUE_WORDS.keys())

                        # Corrupt the sentence
                        corrupted, num_changes = corrupt_sentence_selective(sent)

                        if num_changes == 0:
                            continue

                        if corrupted == sent:
                            continue

                        pair = {'source': corrupted, 'target': sent}

                        if has_high_value:
                            high_value_pairs.append(pair)
                            stats['high_value'] += 1
                        else:
                            pairs.append(pair)
                            stats['regular'] += 1

                        # Count hamza types
                        for char in 'إأآؤئ':
                            if char in sent:
                                stats[f'hamza_{char}'] += 1

                        if len(pairs) + len(high_value_pairs) >= target_pairs * 1.5:
                            break

                    if len(pairs) + len(high_value_pairs) >= target_pairs * 1.5:
                        break

        except Exception as e:
            print(f"  Error processing {file_path}: {e}", flush=True)
            continue

        if len(pairs) + len(high_value_pairs) >= target_pairs * 1.5:
            break

    # Balance: prioritize high-value pairs
    print(f"\nTotal pairs: {len(pairs):,} regular, {len(high_value_pairs):,} high-value", flush=True)

    # Take all high-value pairs, fill rest with regular
    final_pairs = high_value_pairs.copy()
    remaining = target_pairs - len(final_pairs)
    if remaining > 0:
        random.shuffle(pairs)
        final_pairs.extend(pairs[:remaining])

    random.shuffle(final_pairs)

    # Save
    print(f"\nSaving {len(final_pairs):,} pairs to {output_file}", flush=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_pairs, f, ensure_ascii=False, indent=2)

    print("\nStatistics:", flush=True)
    for key, value in sorted(stats.items()):
        print(f"  {key}: {value:,}", flush=True)

    return final_pairs


def generate_single_char_augmentation(output_file, target_pairs=20000):
    """Generate single character substitution and insertion errors."""

    # Common Arabic character confusions
    CONFUSIONS = [
        ('ض', 'ظ'), ('ظ', 'ض'),  # dad/za
        ('د', 'ذ'), ('ذ', 'د'),  # dal/thal
        ('س', 'ص'), ('ص', 'س'),  # seen/sad
        ('ت', 'ط'), ('ط', 'ت'),  # ta/ta
        ('ك', 'ق'), ('ق', 'ك'),  # kaf/qaf
        ('ه', 'ة'), ('ة', 'ه'),  # haa/taa_marbuta
        ('ى', 'ي'), ('ي', 'ى'),  # alif_maqsura/yaa
        ('ا', 'أ'), ('أ', 'ا'),  # alif/hamza_alif
        ('و', 'ؤ'), ('ؤ', 'و'),  # waw/hamza_waw
    ]

    # Common words to corrupt
    WORD_BANK = [
        'الحكومة', 'المدرسة', 'الجامعة', 'المستشفى', 'الشركة',
        'المدينة', 'القرية', 'الدولة', 'الوزارة', 'المؤسسة',
        'الرئيس', 'الوزير', 'المدير', 'الطالب', 'المعلم',
        'الكتاب', 'القلم', 'الورقة', 'الحاسوب', 'الهاتف',
        'السيارة', 'الطائرة', 'القطار', 'الباص', 'السفينة',
        'الماء', 'الهواء', 'النار', 'التراب', 'الشمس',
        'القمر', 'النجوم', 'السماء', 'الأرض', 'البحر',
        'يكتب', 'يقرأ', 'يعمل', 'يدرس', 'يسافر',
        'كبير', 'صغير', 'جديد', 'قديم', 'جميل',
        'سريع', 'بطيء', 'قوي', 'ضعيف', 'طويل',
    ]

    pairs = []

    # Single char substitution
    for _ in range(target_pairs // 2):
        word = random.choice(WORD_BANK)
        orig, repl = random.choice(CONFUSIONS)
        if orig in word:
            corrupted = word.replace(orig, repl, 1)
            if corrupted != word:
                # Create a simple sentence context
                templates = [
                    f"هذا {word} جيد",
                    f"رأيت {word} في المكان",
                    f"{word} مهم جداً",
                    f"أريد {word} الآن",
                    f"هل تعرف {word}",
                ]
                template = random.choice(templates)
                correct_sent = template
                corrupt_sent = template.replace(word, corrupted)
                if corrupt_sent != correct_sent:
                    pairs.append({'source': corrupt_sent, 'target': correct_sent})

    # Single char insertion (extra letter)
    for _ in range(target_pairs // 4):
        word = random.choice(WORD_BANK)
        if len(word) > 3:
            pos = random.randint(1, len(word) - 1)
            # Insert a duplicate letter
            corrupted = word[:pos] + word[pos] + word[pos:]
            templates = [
                f"هذا {word} جيد",
                f"رأيت {word} في المكان",
                f"{word} مهم جداً",
            ]
            template = random.choice(templates)
            correct_sent = template
            corrupt_sent = template.replace(word, corrupted)
            if corrupt_sent != correct_sent:
                pairs.append({'source': corrupt_sent, 'target': correct_sent})

    # Single char deletion
    for _ in range(target_pairs // 4):
        word = random.choice(WORD_BANK)
        if len(word) > 4:
            pos = random.randint(1, len(word) - 2)
            corrupted = word[:pos] + word[pos+1:]
            templates = [
                f"هذا {word} جيد",
                f"رأيت {word} في المكان",
                f"{word} مهم جداً",
            ]
            template = random.choice(templates)
            correct_sent = template
            corrupt_sent = template.replace(word, corrupted)
            if corrupt_sent != correct_sent:
                pairs.append({'source': corrupt_sent, 'target': correct_sent})

    random.shuffle(pairs)

    print(f"Generated {len(pairs):,} single-char augmentation pairs", flush=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)

    return pairs


def process_single_txt_file(corpus_file, output_file, target_pairs=100000):
    """Process a single large MSA corpus file."""

    pairs = []
    high_value_pairs = []
    stats = defaultdict(int)

    print(f"Processing {corpus_file}...", flush=True)

    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            if line_num % 500000 == 0:
                print(f"  {line_num:,} lines, {len(pairs) + len(high_value_pairs):,} pairs", flush=True)

            text = line.strip()
            if not text or len(text) < 15:
                continue

            # Split into sentences
            sentences = re.split(r'[.!؟\n]', text)

            for sent in sentences:
                sent = sent.strip()
                if len(sent) < 15 or len(sent) > 200:
                    continue

                if not has_hamza(sent):
                    continue

                # Check for high-value words
                has_high_value = any(w in sent for w in HIGH_VALUE_WORDS.keys())

                # Corrupt the sentence
                corrupted, num_changes = corrupt_sentence_selective(sent)

                if num_changes == 0:
                    continue

                if corrupted == sent:
                    continue

                pair = {'source': corrupted, 'target': sent}

                if has_high_value:
                    high_value_pairs.append(pair)
                    stats['high_value'] += 1
                else:
                    pairs.append(pair)
                    stats['regular'] += 1

                # Count hamza types
                for char in 'إأآؤئ':
                    if char in sent:
                        stats[f'hamza_{char}'] += 1

                # Early stop if we have enough
                if len(pairs) + len(high_value_pairs) >= target_pairs * 2:
                    break

            if len(pairs) + len(high_value_pairs) >= target_pairs * 2:
                break

    # Balance: prioritize high-value pairs
    print(f"\nTotal pairs: {len(pairs):,} regular, {len(high_value_pairs):,} high-value", flush=True)

    # Take all high-value pairs, fill rest with regular
    final_pairs = high_value_pairs.copy()
    remaining = target_pairs - len(final_pairs)
    if remaining > 0:
        random.shuffle(pairs)
        final_pairs.extend(pairs[:remaining])

    random.shuffle(final_pairs)

    # Save
    print(f"\nSaving {len(final_pairs):,} pairs to {output_file}", flush=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_pairs, f, ensure_ascii=False, indent=2)

    print("\nStatistics:", flush=True)
    for key, value in sorted(stats.items()):
        print(f"  {key}: {value:,}", flush=True)

    return final_pairs


def main():
    print("=" * 70, flush=True)
    print("HAMZA AUGMENTATION DATA GENERATOR", flush=True)
    print("=" * 70, flush=True)

    # Paths - using the actual MSA corpus location
    MSA_CORPUS_FILE = "/home/ubuntu/nahawi/corpus/combined/msa_corpus_full.txt"
    HAMZA_OUTPUT = "/home/ubuntu/nahawi/data/hamza_augmentation.json"
    CHAR_OUTPUT = "/home/ubuntu/nahawi/data/single_char_augmentation.json"

    # Process MSA corpus
    if os.path.exists(MSA_CORPUS_FILE):
        print(f"\nProcessing MSA corpus at {MSA_CORPUS_FILE}...", flush=True)
        hamza_pairs = process_single_txt_file(MSA_CORPUS_FILE, HAMZA_OUTPUT, target_pairs=100000)
    else:
        print(f"\nERROR: MSA corpus not found at {MSA_CORPUS_FILE}!", flush=True)
        return

    print("\n" + "=" * 70, flush=True)
    print("SINGLE CHARACTER AUGMENTATION", flush=True)
    print("=" * 70, flush=True)

    char_pairs = generate_single_char_augmentation(CHAR_OUTPUT, target_pairs=20000)

    print("\n" + "=" * 70, flush=True)
    print("DONE", flush=True)
    print("=" * 70, flush=True)
    print(f"Hamza pairs: {len(hamza_pairs):,}", flush=True)
    print(f"Single-char pairs: {len(char_pairs):,}", flush=True)
    print(f"\nOutput files:", flush=True)
    print(f"  - {HAMZA_OUTPUT}", flush=True)
    print(f"  - {CHAR_OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
