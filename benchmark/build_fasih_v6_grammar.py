#!/usr/bin/env python3
"""
FASIH v6 Grammar Benchmark Builder

Creates comprehensive grammar-specific evaluation using CAMeL Tools.
Target: 100+ samples per category for 5 grammar categories = 500+ grammar samples

Categories:
1. gender_noun_adj     - الطالبات المتفوقين → المتفوقات
2. number_subj_verb    - الموظفون يحتاجوا → يحتاجون
3. verb_gender         - الطالبات حصلوا → حصلن
4. relative_pronoun    - القصة الذي → التي
5. case_ending         - المعلمين يعملون → المعلمون
"""

import json
import random
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# Try to import CAMeL Tools
try:
    from camel_tools.morphology.database import MorphologyDB
    from camel_tools.morphology.analyzer import Analyzer
    CAMEL_AVAILABLE = True
except ImportError:
    CAMEL_AVAILABLE = False
    print("Warning: CAMeL Tools not available. Using pattern-based generation.")


@dataclass
class GrammarSample:
    source: str  # Incorrect
    target: str  # Correct
    error_type: str
    correction: str
    pattern: str  # Sub-pattern for analysis


# ============================================================================
# GRAMMAR PATTERNS DATABASE
# ============================================================================

# Feminine plural nouns (الطالبات، المهندسات، etc.)
FEM_PLURAL_NOUNS = [
    "الطالبات", "المهندسات", "الطبيبات", "المعلمات", "الموظفات",
    "المحاسبات", "الممرضات", "الباحثات", "المدرسات", "الكاتبات",
    "المديرات", "الصحفيات", "المحاميات", "العاملات", "المتخصصات",
    "الفنانات", "الرياضيات", "السياسيات", "الأمهات", "البنات",
]

# Masculine plural nouns (الطلاب، المهندسون، etc.)
MASC_PLURAL_NOUNS = [
    "الطلاب", "المهندسون", "الأطباء", "المعلمون", "الموظفون",
    "المحاسبون", "الممرضون", "الباحثون", "المدرسون", "الكتّاب",
    "المديرون", "الصحفيون", "المحامون", "العاملون", "المتخصصون",
    "الشباب", "الرجال", "الأولاد", "العلماء", "المهندسين",
]

# Adjectives with gender pairs (masc, fem)
ADJECTIVE_PAIRS = [
    ("المتفوقون", "المتفوقات"),
    ("الناجحون", "الناجحات"),
    ("المجتهدون", "المجتهدات"),
    ("الماهرون", "الماهرات"),
    ("الجادون", "الجادات"),
    ("المتميزون", "المتميزات"),
    ("الموهوبون", "الموهوبات"),
    ("المبدعون", "المبدعات"),
    ("الخبراء", "الخبيرات"),
    ("المحترفون", "المحترفات"),
    ("المتفانون", "المتفانيات"),
    ("الكفؤون", "الكفؤات"),
    ("المخلصون", "المخلصات"),
    ("النشيطون", "النشيطات"),
    ("الذكيّون", "الذكيّات"),
]

# Singular adjectives (masc, fem)
SINGULAR_ADJ_PAIRS = [
    ("الكبير", "الكبيرة"),
    ("الصغير", "الصغيرة"),
    ("الجديد", "الجديدة"),
    ("القديم", "القديمة"),
    ("الجميل", "الجميلة"),
    ("السريع", "السريعة"),
    ("الطويل", "الطويلة"),
    ("القصير", "القصيرة"),
    ("الذكي", "الذكية"),
    ("القوي", "القوية"),
]

# Feminine singular nouns
FEM_SINGULAR_NOUNS = [
    "المدينة", "الجامعة", "المدرسة", "الشركة", "الحكومة",
    "القصة", "الرواية", "المقالة", "الدراسة", "المنظمة",
    "السيارة", "الطائرة", "السفينة", "الحافلة", "الدراجة",
    "الغرفة", "الحديقة", "المكتبة", "الساحة", "البناية",
]

# Masculine singular nouns
MASC_SINGULAR_NOUNS = [
    "البيت", "الكتاب", "القلم", "الباب", "الشارع",
    "المكتب", "الفصل", "المتحف", "المسجد", "الملعب",
    "الطفل", "الرجل", "الولد", "الطالب", "المعلم",
    "البحر", "النهر", "الجبل", "السوق", "المطعم",
]

# Verb pairs (masculine plural, feminine plural) - past tense
VERB_PAIRS_PAST = [
    ("حصلوا", "حصلن"),
    ("ذهبوا", "ذهبن"),
    ("كتبوا", "كتبن"),
    ("درسوا", "درسن"),
    ("عملوا", "عملن"),
    ("نجحوا", "نجحن"),
    ("فازوا", "فزن"),
    ("قرأوا", "قرأن"),
    ("سافروا", "سافرن"),
    ("شاركوا", "شاركن"),
    ("صمموا", "صممن"),
    ("قدموا", "قدمن"),
    ("أنجزوا", "أنجزن"),
    ("حققوا", "حققن"),
    ("أبدعوا", "أبدعن"),
]

# Verb pairs (masculine plural, feminine plural) - present tense
VERB_PAIRS_PRESENT = [
    ("يحتاجون", "يحتجن"),
    ("يعملون", "يعملن"),
    ("يدرسون", "يدرسن"),
    ("يكتبون", "يكتبن"),
    ("يبحثون", "يبحثن"),
    ("يسعون", "يسعين"),
    ("يقدمون", "يقدمن"),
    ("يشاركون", "يشاركن"),
    ("يجتهدون", "يجتهدن"),
    ("يتعلمون", "يتعلمن"),
]

# Colloquial verb forms (wrong → correct)
COLLOQUIAL_VERB_CORRECTIONS = [
    ("يحتاجوا", "يحتاجون"),
    ("يعملوا", "يعملون"),
    ("يدرسوا", "يدرسون"),
    ("يكتبوا", "يكتبون"),
    ("يبحثوا", "يبحثون"),
    ("يقدموا", "يقدمون"),
    ("يشاركوا", "يشاركون"),
    ("يدققوا", "يدققون"),
    ("يبذلوا", "يبذلون"),
    ("يسافروا", "يسافرون"),
]

# Relative pronouns
RELATIVE_PRONOUNS = {
    "masc": "الذي",
    "fem": "التي",
    "masc_plural": "الذين",
    "fem_plural": "اللاتي",
}

# Case endings (nominative ون vs accusative/genitive ين)
CASE_PAIRS = [
    ("المعلمون", "المعلمين"),
    ("المهندسون", "المهندسين"),
    ("المحاسبون", "المحاسبين"),
    ("الموظفون", "الموظفين"),
    ("الباحثون", "الباحثين"),
    ("المترجمون", "المترجمين"),
    ("المديرون", "المديرين"),
    ("العاملون", "العاملين"),
    ("المتخصصون", "المتخصصين"),
    ("المشاركون", "المشاركين"),
]


# ============================================================================
# SENTENCE TEMPLATES
# ============================================================================

# Templates for gender agreement (noun + adjective)
GENDER_NOUN_ADJ_TEMPLATES = [
    "{noun} {adj} حصلوا على جوائز في المسابقة الدولية.",
    "كرّمت الجامعة {noun} {adj} على إنجازاتهم العلمية.",
    "يعمل {noun} {adj} في مشاريع بحثية متقدمة.",
    "تخرج {noun} {adj} من أفضل الجامعات العربية.",
    "شارك {noun} {adj} في المؤتمر السنوي بنجاح.",
    "حضر {noun} {adj} الاجتماع الأسبوعي في الشركة.",
    "أنجز {noun} {adj} مهامهم بكفاءة عالية.",
    "تميز {noun} {adj} بإبداعهم في العمل.",
    "فاز {noun} {adj} بالمركز الأول في المنافسة.",
    "قدم {noun} {adj} أفكاراً مبتكرة للمشروع.",
]

# Templates for number agreement (subject + verb)
NUMBER_SUBJ_VERB_TEMPLATES = [
    "{subject} {verb} إلى تدريب مكثف على الأنظمة الحديثة.",
    "{subject} {verb} عن فرص عمل في القطاع الخاص.",
    "{subject} {verb} حلولاً مبتكرة لمشاكل الطاقة.",
    "{subject} {verb} في السجلات المالية بدقة متناهية.",
    "{subject} {verb} جهوداً كبيرة لرفع مستوى الأداء.",
    "{subject} {verb} بجد لإنجاز المشاريع في الوقت المحدد.",
    "{subject} {verb} على تطوير مهاراتهم باستمرار.",
    "{subject} {verb} في الأنشطة الثقافية والاجتماعية.",
    "{subject} {verb} بالعمل الجماعي لتحقيق الأهداف.",
    "{subject} {verb} من الخبرات الدولية في مجالهم.",
]

# Templates for verb gender agreement
VERB_GENDER_TEMPLATES = [
    "{subject} {verb} على منح دراسية من الجامعة.",
    "{subject} {verb} مشروعاً ضخماً يخدم المدينة.",
    "{subject} {verb} جهوداً كبيرة لرفع مستوى الطالبات.",
    "{subject} {verb} على تقديم أفضل رعاية صحية للمرضى.",
    "{subject} {verb} بجد لإنجاز أبحاثهن في الوقت المحدد.",
    "{subject} {verb} إلى الخارج لاستكمال دراستهن.",
    "{subject} {verb} بنجاح في المؤتمر الدولي.",
    "{subject} {verb} جوائز التميز على مستوى الوطن.",
    "{subject} {verb} أرقاماً قياسية في البطولة.",
    "{subject} {verb} إسهامات كبيرة في المجال العلمي.",
]

# Templates for relative pronouns
RELATIVE_PRONOUN_TEMPLATES = [
    "{noun} {rel} قرأتها أمس كانت مؤثرة جداً.",
    "{noun} {rel} شاهدناه كان رائعاً ومفيداً.",
    "{noun} {rel} زرناها تتميز بجمال طبيعتها.",
    "{noun} {rel} درسناها تتضمن معلومات قيمة.",
    "{noun} {rel} حضرناه ناقش قضايا مهمة.",
    "{noun} {rel} اشتريناها تعمل بكفاءة عالية.",
    "{noun} {rel} قابلناه يتميز بخبرة واسعة.",
    "{noun} {rel} أنشأناها تقدم خدمات متميزة.",
    "{noun} {rel} اخترناه يناسب احتياجاتنا.",
    "{noun} {rel} تعلمناها مفيدة جداً في العمل.",
]

# Templates for case endings
CASE_ENDING_TEMPLATES = [
    "{subject} يعملون بجد لتحقيق الأهداف المنشودة.",
    "يشارك {subject} في المؤتمرات الدولية بانتظام.",
    "حضر {subject} الاجتماع السنوي للمنظمة.",
    "كرّمت الوزارة {subject} على إنجازاتهم.",
    "يسعى {subject} لتطوير مهاراتهم باستمرار.",
    "شارك في الندوة عدد من {subject}.",
    "قدم {subject} عروضاً متميزة في المعرض.",
    "التقى الوزير بـ{subject} لمناقشة القضايا.",
    "يتميز {subject} بخبراتهم الواسعة.",
    "أثنى المدير على {subject} لجهودهم.",
]


# ============================================================================
# SAMPLE GENERATORS
# ============================================================================

def generate_gender_noun_adj_samples(count: int = 100) -> List[GrammarSample]:
    """
    Generate samples for gender agreement between noun and adjective.

    Error: Feminine noun + masculine adjective
    Correction: Use feminine adjective

    Example:
    - Wrong:   الطالبات المتفوقين حصلوا على جوائز
    - Correct: الطالبات المتفوقات حصلن على جوائز
    """
    samples = []

    for _ in range(count):
        template = random.choice(GENDER_NOUN_ADJ_TEMPLATES)
        noun = random.choice(FEM_PLURAL_NOUNS)
        masc_adj, fem_adj = random.choice(ADJECTIVE_PAIRS)

        # Source: feminine noun + masculine adjective (WRONG)
        source = template.format(noun=noun, adj=masc_adj)
        # Target: feminine noun + feminine adjective (CORRECT)
        target = template.format(noun=noun, adj=fem_adj)

        samples.append(GrammarSample(
            source=source,
            target=target,
            error_type="gender_noun_adj",
            correction=f"{masc_adj} -> {fem_adj}",
            pattern="fem_noun_masc_adj"
        ))

    # Also generate masculine noun + feminine adjective errors
    for _ in range(count // 2):
        template = random.choice(GENDER_NOUN_ADJ_TEMPLATES)
        noun = random.choice(MASC_PLURAL_NOUNS)
        masc_adj, fem_adj = random.choice(ADJECTIVE_PAIRS)

        # Source: masculine noun + feminine adjective (WRONG)
        source = template.format(noun=noun, adj=fem_adj)
        # Target: masculine noun + masculine adjective (CORRECT)
        target = template.format(noun=noun, adj=masc_adj)

        samples.append(GrammarSample(
            source=source,
            target=target,
            error_type="gender_noun_adj",
            correction=f"{fem_adj} -> {masc_adj}",
            pattern="masc_noun_fem_adj"
        ))

    # Singular noun-adjective agreement
    for _ in range(count // 2):
        template = "{noun} {adj} تجذب الزوار من مختلف أنحاء العالم."
        noun = random.choice(FEM_SINGULAR_NOUNS)
        masc_adj, fem_adj = random.choice(SINGULAR_ADJ_PAIRS)

        source = template.format(noun=noun, adj=masc_adj)
        target = template.format(noun=noun, adj=fem_adj)

        samples.append(GrammarSample(
            source=source,
            target=target,
            error_type="gender_noun_adj",
            correction=f"{masc_adj} -> {fem_adj}",
            pattern="fem_sing_noun_masc_adj"
        ))

    random.shuffle(samples)
    return samples[:count]


def generate_number_subj_verb_samples(count: int = 100) -> List[GrammarSample]:
    """
    Generate samples for number agreement - colloquial verb forms.

    Error: Using colloquial يحتاجوا instead of MSA يحتاجون

    Example:
    - Wrong:   الموظفون يحتاجوا إلى تدريب
    - Correct: الموظفون يحتاجون إلى تدريب
    """
    samples = []

    for _ in range(count):
        template = random.choice(NUMBER_SUBJ_VERB_TEMPLATES)
        subject = random.choice(MASC_PLURAL_NOUNS)
        wrong_verb, correct_verb = random.choice(COLLOQUIAL_VERB_CORRECTIONS)

        source = template.format(subject=subject, verb=wrong_verb)
        target = template.format(subject=subject, verb=correct_verb)

        samples.append(GrammarSample(
            source=source,
            target=target,
            error_type="number_subj_verb",
            correction=f"{wrong_verb} -> {correct_verb}",
            pattern="colloquial_verb_ending"
        ))

    random.shuffle(samples)
    return samples[:count]


def generate_verb_gender_samples(count: int = 100) -> List[GrammarSample]:
    """
    Generate samples for verb gender agreement.

    Error: Feminine subject + masculine verb

    Example:
    - Wrong:   الطالبات حصلوا على منح
    - Correct: الطالبات حصلن على منح
    """
    samples = []

    # Past tense
    for _ in range(count // 2):
        template = random.choice(VERB_GENDER_TEMPLATES)
        subject = random.choice(FEM_PLURAL_NOUNS)
        masc_verb, fem_verb = random.choice(VERB_PAIRS_PAST)

        source = template.format(subject=subject, verb=masc_verb)
        target = template.format(subject=subject, verb=fem_verb)

        samples.append(GrammarSample(
            source=source,
            target=target,
            error_type="verb_gender",
            correction=f"{masc_verb} -> {fem_verb}",
            pattern="fem_subj_masc_verb_past"
        ))

    # Present tense
    for _ in range(count // 2):
        template = "{subject} {verb} على تحقيق أهدافهن بنجاح."
        subject = random.choice(FEM_PLURAL_NOUNS)
        masc_verb, fem_verb = random.choice(VERB_PAIRS_PRESENT)

        source = template.format(subject=subject, verb=masc_verb)
        target = template.format(subject=subject, verb=fem_verb)

        samples.append(GrammarSample(
            source=source,
            target=target,
            error_type="verb_gender",
            correction=f"{masc_verb} -> {fem_verb}",
            pattern="fem_subj_masc_verb_present"
        ))

    random.shuffle(samples)
    return samples[:count]


def generate_relative_pronoun_samples(count: int = 100) -> List[GrammarSample]:
    """
    Generate samples for relative pronoun agreement.

    Error: Feminine noun + masculine relative pronoun

    Example:
    - Wrong:   القصة الذي قرأتها
    - Correct: القصة التي قرأتها
    """
    samples = []

    for _ in range(count):
        template = random.choice(RELATIVE_PRONOUN_TEMPLATES)
        noun = random.choice(FEM_SINGULAR_NOUNS)

        # Wrong: feminine noun + الذي
        source = template.format(noun=noun, rel="الذي")
        # Correct: feminine noun + التي
        target = template.format(noun=noun, rel="التي")

        samples.append(GrammarSample(
            source=source,
            target=target,
            error_type="relative_pronoun",
            correction="الذي -> التي",
            pattern="fem_noun_masc_rel"
        ))

    # Also masculine noun + feminine relative pronoun
    for _ in range(count // 2):
        template = "{noun} {rel} قرأناه كان مفيداً جداً."
        noun = random.choice(MASC_SINGULAR_NOUNS)

        source = template.format(noun=noun, rel="التي")
        target = template.format(noun=noun, rel="الذي")

        samples.append(GrammarSample(
            source=source,
            target=target,
            error_type="relative_pronoun",
            correction="التي -> الذي",
            pattern="masc_noun_fem_rel"
        ))

    random.shuffle(samples)
    return samples[:count]


def generate_case_ending_samples(count: int = 100) -> List[GrammarSample]:
    """
    Generate samples for case ending errors.

    Error: Using accusative/genitive (ين) where nominative (ون) is required

    Example:
    - Wrong:   المعلمين يعملون بجد (subject should be nominative)
    - Correct: المعلمون يعملون بجد
    """
    samples = []

    for _ in range(count):
        template = random.choice(CASE_ENDING_TEMPLATES)
        nom_form, acc_form = random.choice(CASE_PAIRS)

        # Wrong: accusative where nominative needed (as subject)
        source = template.format(subject=acc_form)
        target = template.format(subject=nom_form)

        samples.append(GrammarSample(
            source=source,
            target=target,
            error_type="case_ending",
            correction=f"{acc_form} -> {nom_form}",
            pattern="acc_for_nom"
        ))

    random.shuffle(samples)
    return samples[:count]


# ============================================================================
# MAIN BUILDER
# ============================================================================

def build_fasih_v6(output_dir: Path, samples_per_category: int = 100):
    """Build FASIH v6 grammar benchmark."""

    print("=" * 60)
    print("FASIH v6 Grammar Benchmark Builder")
    print("=" * 60)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate samples for each category
    print("\nGenerating samples...")

    all_samples = []
    category_counts = {}

    # 1. Gender agreement (noun + adjective)
    print(f"  [1/5] Gender Noun-Adj: {samples_per_category} samples")
    gender_samples = generate_gender_noun_adj_samples(samples_per_category)
    all_samples.extend(gender_samples)
    category_counts["gender_noun_adj"] = len(gender_samples)

    # 2. Number agreement (colloquial verb forms)
    print(f"  [2/5] Number Subj-Verb: {samples_per_category} samples")
    number_samples = generate_number_subj_verb_samples(samples_per_category)
    all_samples.extend(number_samples)
    category_counts["number_subj_verb"] = len(number_samples)

    # 3. Verb gender agreement
    print(f"  [3/5] Verb Gender: {samples_per_category} samples")
    verb_samples = generate_verb_gender_samples(samples_per_category)
    all_samples.extend(verb_samples)
    category_counts["verb_gender"] = len(verb_samples)

    # 4. Relative pronouns
    print(f"  [4/5] Relative Pronouns: {samples_per_category} samples")
    rel_samples = generate_relative_pronoun_samples(samples_per_category)
    all_samples.extend(rel_samples)
    category_counts["relative_pronoun"] = len(rel_samples)

    # 5. Case endings
    print(f"  [5/5] Case Endings: {samples_per_category} samples")
    case_samples = generate_case_ending_samples(samples_per_category)
    all_samples.extend(case_samples)
    category_counts["case_ending"] = len(case_samples)

    # Shuffle all samples
    random.shuffle(all_samples)

    # Split into test (80%) and dev (20%)
    split_idx = int(len(all_samples) * 0.8)
    test_samples = all_samples[:split_idx]
    dev_samples = all_samples[split_idx:]

    # Convert to JSON format
    def to_json(sample: GrammarSample) -> dict:
        return {
            "source": sample.source,
            "target": sample.target,
            "error_type": sample.error_type,
            "correction": sample.correction,
            "pattern": sample.pattern
        }

    test_json = [to_json(s) for s in test_samples]
    dev_json = [to_json(s) for s in dev_samples]

    # Create rubric
    rubric = {
        "version": "6.0",
        "description": "FASIH v6 - Grammar-Focused Arabic GEC Benchmark",
        "purpose": "Evaluate grammar agreement capabilities specifically",
        "categories": {
            "gender_noun_adj": {
                "arabic": "مطابقة الجنس (اسم + صفة)",
                "description": "Noun-adjective gender agreement",
                "examples": ["الطالبات المتفوقين → المتفوقات"],
                "count": category_counts["gender_noun_adj"]
            },
            "number_subj_verb": {
                "arabic": "مطابقة العدد (فاعل + فعل)",
                "description": "Subject-verb number agreement, colloquial verb forms",
                "examples": ["يحتاجوا → يحتاجون"],
                "count": category_counts["number_subj_verb"]
            },
            "verb_gender": {
                "arabic": "مطابقة جنس الفعل",
                "description": "Verb-subject gender agreement",
                "examples": ["الطالبات حصلوا → حصلن"],
                "count": category_counts["verb_gender"]
            },
            "relative_pronoun": {
                "arabic": "الاسم الموصول",
                "description": "Relative pronoun gender agreement",
                "examples": ["القصة الذي → التي"],
                "count": category_counts["relative_pronoun"]
            },
            "case_ending": {
                "arabic": "علامة الإعراب",
                "description": "Case ending (nominative vs accusative/genitive)",
                "examples": ["المعلمين يعملون → المعلمون"],
                "count": category_counts["case_ending"]
            }
        },
        "statistics": {
            "total_test": len(test_json),
            "total_dev": len(dev_json),
            "total": len(all_samples),
            "categories": 5,
            "by_category": category_counts
        },
        "evaluation": {
            "metric": "F0.5 per category",
            "success_threshold": {
                "minimum": 50,
                "target": 70,
                "excellent": 85
            }
        }
    }

    # Write files
    test_path = output_dir / "fasih_v6_grammar_test.json"
    dev_path = output_dir / "fasih_v6_grammar_dev.json"
    rubric_path = output_dir / "fasih_v6_grammar_rubric.json"

    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_json, f, ensure_ascii=False, indent=2)

    with open(dev_path, "w", encoding="utf-8") as f:
        json.dump(dev_json, f, ensure_ascii=False, indent=2)

    with open(rubric_path, "w", encoding="utf-8") as f:
        json.dump(rubric, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("FASIH v6 Grammar Benchmark Generated")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")
    print(f"\nFiles created:")
    print(f"  - {test_path.name}: {len(test_json)} samples")
    print(f"  - {dev_path.name}: {len(dev_json)} samples")
    print(f"  - {rubric_path.name}")
    print(f"\nCategory breakdown:")
    for cat, count in category_counts.items():
        print(f"  - {cat}: {count}")
    print(f"\nTotal: {len(all_samples)} samples")

    return test_json, dev_json, rubric


if __name__ == "__main__":
    output_dir = Path(__file__).parent / "fasih_v6_grammar"
    samples = 100  # Per category

    if len(sys.argv) > 1:
        samples = int(sys.argv[1])

    build_fasih_v6(output_dir, samples_per_category=samples)
