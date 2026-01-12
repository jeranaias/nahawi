#!/usr/bin/env python3
"""
Hunt through the MSA corpus to find more samples for underrepresented categories.

This script:
1. Searches the corpus for words that commonly have errors
2. Creates synthetic error pairs from correct text
3. Balances all categories to ~150 samples each
"""

import json
import random
import re
import sys
from pathlib import Path
from collections import defaultdict

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).parent.parent
MSA_CORPUS = PROJECT_ROOT / "corpus" / "msa_corpus_full.txt"
FASIH_DIR = Path(__file__).parent / "fasih"

# Common error patterns - correct form : error forms
HAMZA_PATTERNS = {
    # Word-initial hamza with fatha (أ)
    'أعلن': ['اعلن'], 'أكثر': ['اكثر'], 'أول': ['اول'],
    'أخير': ['اخير'], 'أمام': ['امام'], 'أحد': ['احد'],
    'أصبح': ['اصبح'], 'أخذ': ['اخذ'], 'أمر': ['امر'],
    'أهم': ['اهم'], 'أخرى': ['اخرى'], 'أيضا': ['ايضا'],
    'أساس': ['اساس'], 'أصل': ['اصل'], 'أثناء': ['اثناء'],
    'أجل': ['اجل'], 'أمن': ['امن'], 'أجنبي': ['اجنبي'],
    # Word-initial hamza with kasra (إ)
    'إلى': ['الى'], 'إن': ['ان'], 'إنه': ['انه'],
    'إذا': ['اذا'], 'إنها': ['انها'], 'إعلان': ['اعلان'],
    'إنشاء': ['انشاء'], 'إطار': ['اطار'], 'إجراء': ['اجراء'],
    'إضافة': ['اضافة'], 'إمكانية': ['امكانية'], 'إعادة': ['اعادة'],
    'إصدار': ['اصدار'], 'إقامة': ['اقامة'], 'إدارة': ['ادارة'],
}

ALIF_MADDA_PATTERNS = {
    'الآن': ['الان'], 'آخر': ['اخر'], 'الآخر': ['الاخر'],
    'القرآن': ['القران'], 'الآلات': ['الالات'], 'آلة': ['الة'],
    'آسيا': ['اسيا'], 'آثار': ['اثار'], 'الآثار': ['الاثار'],
    'آمن': ['امن'], 'الآمن': ['الامن'], 'آلاف': ['الاف'],
    'آراء': ['اراء'], 'الآراء': ['الاراء'], 'آية': ['اية'],
    'آداب': ['اداب'], 'الآداب': ['الاداب'], 'آفاق': ['افاق'],
}

TAA_MARBUTA_PATTERNS = {
    'المدرسة': 'المدرسه', 'الحكومة': 'الحكومه', 'الدولة': 'الدوله',
    'المنطقة': 'المنطقه', 'الشركة': 'الشركه', 'المدينة': 'المدينه',
    'الجامعة': 'الجامعه', 'المرحلة': 'المرحله', 'الفترة': 'الفتره',
    'السنة': 'السنه', 'اللجنة': 'اللجنه', 'المنظمة': 'المنظمه',
    'الحركة': 'الحركه', 'القضية': 'القضيه', 'الهيئة': 'الهيئه',
    'التجربة': 'التجربه', 'الثقافة': 'الثقافه', 'البيئة': 'البيئه',
    'جديدة': 'جديده', 'كبيرة': 'كبيره', 'صغيرة': 'صغيره',
    'مختلفة': 'مختلفه', 'خاصة': 'خاصه', 'عامة': 'عامه',
}

ALIF_MAQSURA_PATTERNS = {
    'على': 'علي', 'إلى': 'الي', 'حتى': 'حتي',
    'متى': 'متي', 'لدى': 'لدي', 'سوى': 'سوي',
    'مدى': 'مدي', 'أدى': 'ادي', 'معنى': 'معني',
    'مستوى': 'مستوي', 'محتوى': 'محتوي', 'مبنى': 'مبني',
    'الأولى': 'الاولي', 'الكبرى': 'الكبري', 'الأخرى': 'الاخري',
}

DAL_THAL_PATTERNS = {
    'هذا': 'هدا', 'هذه': 'هده', 'إذا': 'ادا',
    'كذلك': 'كدلك', 'لذلك': 'لدلك', 'ذلك': 'دلك',
    'الذي': 'الدي', 'التي': 'الدي', 'منذ': 'مند',
    'إذ': 'اد', 'ذات': 'دات', 'هكذا': 'هكدا',
}

DAD_ZA_PATTERNS = {
    'نظر': 'نضر', 'ظهر': 'ضهر', 'نظام': 'نضام',
    'منظمة': 'منضمة', 'حظر': 'حضر', 'انتظار': 'انتضار',
    'نظرية': 'نضرية', 'منظور': 'منضور', 'ظروف': 'ضروف',
    'عظيم': 'عضيم', 'تنظيم': 'تنضيم', 'محظور': 'محضور',
    'ظل': 'ضل', 'ظلم': 'ضلم', 'مظلوم': 'مضلوم',
}

# Preposition patterns (correct verb + prep combinations)
PREP_PATTERNS = {
    'يعتمد على': 'يعتمد من',
    'يبحث عن': 'يبحث',  # missing
    'يشارك في': 'يشارك ب',
    'يتحدث عن': 'يتحدث',  # missing
    'يهتم ب': 'يهتم',  # missing
    'يعمل على': 'يعمل',  # missing
    'يساعد على': 'يساعد',  # missing
    'يؤثر على': 'يؤثر في',
    'يحتاج إلى': 'يحتاج',  # missing
    'ينتمي إلى': 'ينتمي',  # missing
}


def search_corpus_for_pattern(pattern: str, max_results: int = 50) -> list:
    """Search the corpus for sentences containing a pattern."""
    results = []
    pattern_re = re.compile(r'\b' + re.escape(pattern) + r'\b')

    with open(MSA_CORPUS, 'r', encoding='utf-8') as f:
        for line in f:
            if len(results) >= max_results:
                break
            line = line.strip()
            if pattern_re.search(line) and 20 <= len(line) <= 300:
                results.append(line)

    return results


def create_error_sample(sentence: str, correct: str, error: str, category: str, idx: int) -> dict:
    """Create an error sample by replacing correct form with error form."""
    source = sentence.replace(correct, error)
    if source == sentence:  # No replacement happened
        return None

    return {
        'id': f"core-{category}-{idx:04d}",
        'source': source,
        'target': sentence,
        'category': category,
        'correction': f"{error} → {correct}",
        'source_corpus': 'msa_corpus',
        'difficulty': 'easy' if len(sentence.split()) < 15 else 'medium'
    }


def hunt_hamza_samples(target_count: int = 150) -> list:
    """Hunt for hamza error samples."""
    print(f"\n=== Hunting HAMZA samples (target: {target_count}) ===")
    samples = []

    for correct, errors in HAMZA_PATTERNS.items():
        if len(samples) >= target_count:
            break

        sentences = search_corpus_for_pattern(correct, max_results=20)
        print(f"  {correct}: found {len(sentences)} sentences")

        for sent in sentences:
            if len(samples) >= target_count:
                break
            for error in errors:
                sample = create_error_sample(sent, correct, error, 'hamza', len(samples))
                if sample:
                    samples.append(sample)
                    break

    print(f"  Total hamza: {len(samples)}")
    return samples


def hunt_alif_madda_samples(target_count: int = 150) -> list:
    """Hunt for alif madda error samples."""
    print(f"\n=== Hunting ALIF_MADDA samples (target: {target_count}) ===")
    samples = []

    for correct, errors in ALIF_MADDA_PATTERNS.items():
        if len(samples) >= target_count:
            break

        sentences = search_corpus_for_pattern(correct, max_results=20)
        print(f"  {correct}: found {len(sentences)} sentences")

        for sent in sentences:
            if len(samples) >= target_count:
                break
            for error in errors:
                sample = create_error_sample(sent, correct, error, 'alif_madda', len(samples))
                if sample:
                    samples.append(sample)
                    break

    print(f"  Total alif_madda: {len(samples)}")
    return samples


def hunt_taa_marbuta_samples(target_count: int = 150) -> list:
    """Hunt for taa marbuta error samples."""
    print(f"\n=== Hunting TAA_MARBUTA samples (target: {target_count}) ===")
    samples = []

    for correct, error in TAA_MARBUTA_PATTERNS.items():
        if len(samples) >= target_count:
            break

        sentences = search_corpus_for_pattern(correct, max_results=15)
        print(f"  {correct}: found {len(sentences)} sentences")

        for sent in sentences:
            if len(samples) >= target_count:
                break
            sample = create_error_sample(sent, correct, error, 'taa_marbuta', len(samples))
            if sample:
                samples.append(sample)

    print(f"  Total taa_marbuta: {len(samples)}")
    return samples


def hunt_alif_maqsura_samples(target_count: int = 150) -> list:
    """Hunt for alif maqsura error samples."""
    print(f"\n=== Hunting ALIF_MAQSURA samples (target: {target_count}) ===")
    samples = []

    for correct, error in ALIF_MAQSURA_PATTERNS.items():
        if len(samples) >= target_count:
            break

        sentences = search_corpus_for_pattern(correct, max_results=20)
        print(f"  {correct}: found {len(sentences)} sentences")

        for sent in sentences:
            if len(samples) >= target_count:
                break
            sample = create_error_sample(sent, correct, error, 'alif_maqsura', len(samples))
            if sample:
                samples.append(sample)

    print(f"  Total alif_maqsura: {len(samples)}")
    return samples


def hunt_dal_thal_samples(target_count: int = 150) -> list:
    """Hunt for dal/thal error samples."""
    print(f"\n=== Hunting DAL_THAL samples (target: {target_count}) ===")
    samples = []

    for correct, error in DAL_THAL_PATTERNS.items():
        if len(samples) >= target_count:
            break

        sentences = search_corpus_for_pattern(correct, max_results=20)
        print(f"  {correct}: found {len(sentences)} sentences")

        for sent in sentences:
            if len(samples) >= target_count:
                break
            sample = create_error_sample(sent, correct, error, 'dal_thal', len(samples))
            if sample:
                samples.append(sample)

    print(f"  Total dal_thal: {len(samples)}")
    return samples


def hunt_dad_za_samples(target_count: int = 150) -> list:
    """Hunt for dad/za error samples."""
    print(f"\n=== Hunting DAD_ZA samples (target: {target_count}) ===")
    samples = []

    for correct, error in DAD_ZA_PATTERNS.items():
        if len(samples) >= target_count:
            break

        sentences = search_corpus_for_pattern(correct, max_results=20)
        print(f"  {correct}: found {len(sentences)} sentences")

        for sent in sentences:
            if len(samples) >= target_count:
                break
            sample = create_error_sample(sent, correct, error, 'dad_za', len(samples))
            if sample:
                samples.append(sample)

    print(f"  Total dad_za: {len(samples)}")
    return samples


def hunt_missing_prep_samples(target_count: int = 100) -> list:
    """Hunt for missing preposition samples."""
    print(f"\n=== Hunting MISSING_PREP samples (target: {target_count}) ===")
    samples = []

    for correct_phrase, error_phrase in PREP_PATTERNS.items():
        if len(samples) >= target_count:
            break

        # Only process missing preposition cases
        if error_phrase in correct_phrase:
            continue

        sentences = search_corpus_for_pattern(correct_phrase, max_results=15)
        print(f"  '{correct_phrase}': found {len(sentences)} sentences")

        for sent in sentences:
            if len(samples) >= target_count:
                break
            sample = create_error_sample(sent, correct_phrase, error_phrase, 'missing_prep', len(samples))
            if sample:
                samples.append(sample)

    print(f"  Total missing_prep: {len(samples)}")
    return samples


def main():
    print("=" * 60)
    print("HUNTING FOR PERFECT FASIH SAMPLES")
    print("=" * 60)

    if not MSA_CORPUS.exists():
        print(f"ERROR: MSA corpus not found at {MSA_CORPUS}")
        return

    # Hunt for each category
    all_samples = []

    hamza = hunt_hamza_samples(150)
    all_samples.extend(hamza)

    alif_madda = hunt_alif_madda_samples(150)
    all_samples.extend(alif_madda)

    taa_marbuta = hunt_taa_marbuta_samples(150)
    all_samples.extend(taa_marbuta)

    alif_maqsura = hunt_alif_maqsura_samples(150)
    all_samples.extend(alif_maqsura)

    dal_thal = hunt_dal_thal_samples(150)
    all_samples.extend(dal_thal)

    dad_za = hunt_dad_za_samples(150)
    all_samples.extend(dad_za)

    missing_prep = hunt_missing_prep_samples(100)
    all_samples.extend(missing_prep)

    # Summary
    print("\n" + "=" * 60)
    print("HUNT COMPLETE")
    print("=" * 60)

    from collections import Counter
    cats = Counter(s['category'] for s in all_samples)
    for cat, count in sorted(cats.items()):
        print(f"  {cat}: {count}")
    print(f"\n  TOTAL: {len(all_samples)}")

    # Save hunted samples
    output_file = FASIH_DIR / "hunted_samples.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {output_file}")


if __name__ == "__main__":
    main()
