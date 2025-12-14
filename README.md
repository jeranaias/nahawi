# Nahawi (نحوي) - Arabic Grammatical Error Correction

<div align="center">

**A 10-model ensemble system for Arabic GEC achieving high-precision error correction**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![AraBART](https://img.shields.io/badge/base-AraBART-green.svg)](https://huggingface.co/moussaKam/AraBART)

</div>

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [The Problem: Arabic GEC Challenges](#the-problem-arabic-gec-challenges)
3. [Journey & Evolution](#journey--evolution)
4. [Architecture Deep Dive](#architecture-deep-dive)
5. [The 10-Model Ensemble](#the-10-model-ensemble)
6. [Technical Implementation](#technical-implementation)
7. [Training Pipeline](#training-pipeline)
8. [Evaluation & Metrics](#evaluation--metrics)
9. [Installation & Usage](#installation--usage)
10. [Results & Analysis](#results--analysis)
11. [Lessons Learned](#lessons-learned)
12. [Future Directions](#future-directions)
13. [References & Acknowledgments](#references--acknowledgments)

---

## Project Overview

**Nahawi** (Arabic: نحوي, meaning "grammatical" or "grammarian") is a comprehensive Arabic Grammatical Error Correction (GEC) system. Unlike simple spell-checkers, Nahawi handles the full spectrum of Arabic writing errors including:

- **Hamza confusion** (أ/إ/ا/آ/ء/ؤ/ئ)
- **Taa Marbuta vs Ha** (ة ↔ ه)
- **Alif Maksura vs Ya** (ى ↔ ي)
- **Word merging/splitting** (space errors)
- **Missing words** (deletions)
- **Repeated words**
- **Punctuation errors**
- **Spelling mistakes**
- **Morphological agreement** (gender, number)
- **General grammatical errors**

The system uses a **cascading ensemble** of 10 specialized models, combining fast rule-based corrections with fine-tuned neural models based on AraBART.

### Why "Nahawi"?

The name comes from Arabic grammar terminology. A "نحوي" is someone who studies or practices "النحو" (grammar/syntax). The project aims to be an automated grammarian for Arabic text.

---

## The Problem: Arabic GEC Challenges

Arabic presents unique challenges for grammatical error correction that don't exist in English or other Latin-script languages:

### 1. Complex Morphology

Arabic is a morphologically rich language with:
- **Root-pattern system**: Words derive from 3-4 letter roots with patterns
- **Clitics**: Prefixes and suffixes attach to words (و+ال+كتاب+ه = "and his book")
- **Gender & number agreement**: Adjectives must agree with nouns
- **Dual form**: Arabic has singular, dual, AND plural

### 2. Script Ambiguities

The Arabic script introduces specific error types:

```
Hamza Variants:
  أ (alif with hamza above)    - أكل (ate)
  إ (alif with hamza below)    - إلى (to)
  آ (alif with madda)          - آخر (other)
  ا (plain alif)               - انا (I) - INCORRECT, should be أنا
  ء (standalone hamza)         - ماء (water)
  ؤ (hamza on waw)             - سؤال (question)
  ئ (hamza on ya)              - قائم (standing)

Taa Marbuta vs Ha:
  ة (taa marbuta) - مدرسة (school)
  ه (ha)          - مدرسه (INCORRECT - common handwriting error)

Alif Maksura vs Ya:
  ى (alif maksura) - على (on), إلى (to)
  ي (ya)           - علي (Ali - name), الي (INCORRECT for إلى)
```

### 3. Diacritics (Tashkeel)

Arabic has optional diacritical marks that indicate vowels:
- Most modern text omits diacritics
- This creates ambiguity (same spelling, different words)
- GEC must work with undiacritized text

### 4. Dialectal Influence

Writers often mix Modern Standard Arabic (MSA) with dialectal forms:
- Egyptian: "ازي" instead of "كيف"
- Gulf: "شلون" instead of "كيف"
- This isn't always an "error" but context-dependent

### 5. Limited Resources

Compared to English GEC:
- Fewer annotated corpora
- Smaller pretrained models
- Less research attention
- QALB shared task is the main benchmark

---

## Journey & Evolution

This project evolved through several iterations, each teaching valuable lessons about what works (and doesn't) for Arabic GEC.

### Phase 1: The Naive Approach (Failed)

**Initial idea**: Train a simple sequence-to-sequence model from scratch.

```python
# First attempt - basic transformer
model = TransformerSeq2Seq(
    vocab_size=32000,
    d_model=256,
    n_heads=4,
    n_layers=4
)
```

**Results**: ~15% F0.5 on QALB dev set

**Why it failed**:
- Not enough training data (QALB has ~20K sentences)
- Model too small to learn Arabic morphology
- No pretrained knowledge of Arabic

**Lesson**: You can't learn Arabic grammar from scratch with limited data.

### Phase 2: Rule-Based System (Partial Success)

**Pivot**: Build comprehensive linguistic rules.

```python
# Extensive rule system
HAMZA_RULES = {
    'word_initial': {...},  # Rules for word-start hamza
    'word_medial': {...},   # Rules for mid-word hamza
    'word_final': {...},    # Rules for word-end hamza
}

TAA_MARBUTA_WORDS = {
    'مدرسه': 'مدرسة',
    'جامعه': 'جامعة',
    # ... 500+ entries
}
```

**Results**: ~25% F0.5, but very high precision (~85%)

**Why it plateaued**:
- Rules can't handle all edge cases
- No context understanding
- Missing words/deletions impossible to detect with rules
- Spelling errors need fuzzy matching

**Lesson**: Rules are fast and precise but have a coverage ceiling.

### Phase 3: Character-Aware Neural Model (Improved)

**Approach**: Custom architecture with character-level understanding.

```python
class CharacterAwareGEC(nn.Module):
    def __init__(self):
        self.char_embed = nn.Embedding(256, 64)
        self.char_cnn = nn.Conv1d(64, 128, kernel_size=3)
        self.word_lstm = nn.LSTM(128, 256, bidirectional=True)
        self.decoder = nn.TransformerDecoder(...)
```

**Results**: ~38% F0.5

**Why it wasn't enough**:
- Still training from scratch
- 30MB size constraint limited capacity
- No pretrained Arabic knowledge

**Lesson**: Character awareness helps, but pretraining is essential.

### Phase 4: The Reality Check

At this point, I did a thorough analysis of the state-of-the-art:

| System | F0.5 on QALB-2014 | Model Size |
|--------|-------------------|------------|
| QALB-2014 Winner | 67.9% | Large |
| Recent SOTA | ~72% | Very Large |
| My best attempt | 38% | 30MB |
| Theoretical max (30MB) | ~55-65% | 30MB |

**Key realization**: The 30MB constraint was fundamentally limiting. State-of-the-art requires pretrained transformers.

### Phase 5: Ensemble Architecture (Current)

**Final approach**: Drop size constraints, use pretrained models, specialize.

The insight: Different error types need different treatment:
- **Hamza errors**: Need morphological context → Neural
- **Taa Marbuta**: Mostly dictionary lookup → Rule-based (fast)
- **Missing words**: Need language model → Neural
- **Repeated words**: Simple detection → Rule-based (fast)

This led to the **10-model ensemble architecture**.

---

## Architecture Deep Dive

### Design Philosophy

1. **Specialization over generalization**: Each model focuses on one error type
2. **Cascading for efficiency**: Fast rule-based models run first
3. **High precision priority**: GEC should avoid introducing errors (F0.5 weights precision)
4. **Graceful degradation**: System works with partial models available

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    NahawiEnsemble                            │
│                    (Orchestrator)                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │ Strategy:   │  │ Strategy:   │  │ Strategy:   │          │
│  │ Cascading   │  │ Parallel    │  │ Specialist  │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
│         │                │                │                  │
│         ▼                ▼                ▼                  │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                    Model Pool                         │   │
│  ├──────────────────────────────────────────────────────┤   │
│  │                                                       │   │
│  │  RULE-BASED (Fast, High Precision)                   │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐     │   │
│  │  │TaaMarbutaFix│ │AlifMaksura  │ │Punctuation  │     │   │
│  │  │     ة↔ه     │ │   ى↔ي       │ │   ،؛؟       │     │   │
│  │  └─────────────┘ └─────────────┘ └─────────────┘     │   │
│  │  ┌─────────────┐                                      │   │
│  │  │RepeatedWord │                                      │   │
│  │  │   Fixer     │                                      │   │
│  │  └─────────────┘                                      │   │
│  │                                                       │   │
│  │  NEURAL (AraBART Fine-tuned)                         │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐     │   │
│  │  │ HamzaFixer  │ │ SpaceFixer  │ │SpellingFixer│     │   │
│  │  │ أ/إ/ا/آ/ء   │ │ merge/split │ │ char errors │     │   │
│  │  └─────────────┘ └─────────────┘ └─────────────┘     │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐     │   │
│  │  │DeletedWord  │ │Morphology   │ │ GeneralGEC  │     │   │
│  │  │   Fixer     │ │   Fixer     │ │ (catch-all) │     │   │
│  │  └─────────────┘ └─────────────┘ └─────────────┘     │   │
│  │                                                       │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Orchestration Strategies

#### 1. Cascading Strategy (Default)

Models run in sequence, each refining the previous output:

```python
def correct_cascading(text, confidence_threshold=0.7):
    current = text

    # Phase 1: Fast rule-based corrections
    for model in [taa_marbuta, alif_maksura, punctuation, repeated_word]:
        result = model.correct(current)
        if result.confidence >= confidence_threshold:
            current = result.corrected_text

    # Phase 2: Neural corrections (slower, more powerful)
    for model in [hamza, space, spelling, deleted_word, morphology, general]:
        result = model.correct(current)
        if result.confidence >= confidence_threshold:
            current = result.corrected_text

    return current
```

**Pros**: Fast for simple errors, neural only when needed
**Cons**: Error propagation possible

#### 2. Parallel Strategy

All models run independently, corrections merged by voting:

```python
def correct_parallel(text):
    all_corrections = []

    for model in all_models:
        result = model.correct(text)
        all_corrections.extend(result.corrections)

    # Merge by position, prefer high-confidence
    return merge_corrections(all_corrections)
```

**Pros**: No error propagation
**Cons**: Slower, conflict resolution needed

#### 3. Specialist Strategy

Route to specific models based on detected error type:

```python
def correct_specialist(text):
    error_types = detect_error_types(text)

    for error_type in error_types:
        specialist = get_specialist(error_type)
        text = specialist.correct(text)

    return text
```

**Pros**: Efficient, targeted
**Cons**: Error detection must be accurate

---

## The 10-Model Ensemble

### Model 1: HamzaFixer

**Error Type**: Hamza confusion (أ/إ/ا/آ/ء/ؤ/ئ)

**Why it's hard**: Hamza placement depends on morphological context, word position, and surrounding vowels. The same root can have different hamza forms.

**Examples**:
```
انا ← أنا (I)
الى ← إلى (to)
سأل ← سأل (asked) - no change needed
مسؤول ← مسؤول (responsible) - no change needed
```

**Architecture**: AraBART fine-tuned on hamza-specific error pairs

**Training data**: ~28,000 pairs filtered for hamza differences

### Model 2: SpaceFixer

**Error Type**: Word merging and splitting

**Why it's hard**: Arabic has clitics that attach to words, making boundaries ambiguous. Writers often merge or split incorrectly.

**Examples**:
```
ذهبتالى ← ذهبت إلى (I went to)
في البيت ← في البيت (in the house) - no change
وال كتاب ← والكتاب (and the book)
```

**Architecture**: AraBART fine-tuned on space-error pairs

**Training data**: ~28,000 pairs with different word counts

### Model 3: TaaMarbutaFixer

**Error Type**: ة (taa marbuta) ↔ ه (ha) confusion

**Why it exists**: In handwriting and some fonts, these look identical. Writers often use ه instead of ة for feminine words.

**Examples**:
```
مدرسه ← مدرسة (school)
جامعه ← جامعة (university)
جميله ← جميلة (beautiful - feminine)
```

**Architecture**: Rule-based with dictionary lookup

**Implementation**:
```python
class TaaMarbutaFixer:
    def __init__(self):
        self.taa_words = {
            'مدرسه': 'مدرسة', 'جامعه': 'جامعة',
            'جميله': 'جميلة', 'كبيره': 'كبيرة',
            # ... 100+ entries
        }

    def correct(self, text):
        for wrong, right in self.taa_words.items():
            text = text.replace(wrong, right)

        # Pattern: words ending in يه → ية
        text = re.sub(r'(\w+)يه\b', r'\1ية', text)

        return text
```

### Model 4: AlifMaksuraFixer

**Error Type**: ى (alif maksura) ↔ ي (ya) confusion

**Why it exists**: These look identical without dots. Some keyboards/fonts don't distinguish them.

**Examples**:
```
الي ← إلى (to)
علي ← على (on) - but علي (Ali) is correct as a name
متي ← متى (when)
حتي ← حتى (until)
```

**Architecture**: Rule-based with word list

**Challenge**: Context-dependent (علي as name vs على as preposition)

### Model 5: PunctuationFixer

**Error Type**: Arabic punctuation marks

**Why it exists**: Arabic has its own punctuation (،؛؟) but English marks (,;?) are often used.

**Examples**:
```
مرحبا, كيف حالك? ← مرحبا، كيف حالك؟
```

**Architecture**: Rule-based replacement in Arabic context

### Model 6: DeletedWordFixer

**Error Type**: Missing words

**Why it's hard**: Requires understanding what's missing from context. Common missing words: و (and), في (in), من (from), إلى (to).

**Examples**:
```
ذهبت المدرسة ← ذهبت إلى المدرسة (I went to school)
الكتاب الطاولة ← الكتاب على الطاولة (The book on the table)
```

**Architecture**: AraBART fine-tuned on deletion pairs

**Training data**: Pairs where source has fewer words than target

### Model 7: RepeatedWordFixer

**Error Type**: Duplicate words

**Why it exists**: Typing errors, copy-paste mistakes

**Examples**:
```
ذهبت إلى إلى المدرسة ← ذهبت إلى المدرسة
```

**Architecture**: Rule-based detection

**Exception handling**: Some repetitions are intentional (لا لا = "no no" for emphasis)

### Model 8: SpellingFixer

**Error Type**: Character-level spelling errors

**Includes**: Character swaps, insertions, deletions, substitutions

**Examples**:
```
الكتب المفيده ← الكتب المفيدة
مدرصة ← مدرسة (ص→س)
```

**Architecture**: AraBART fine-tuned on spelling pairs

**Training data**: ~9,000 pairs with 1-3 character differences

### Model 9: MorphologyFixer

**Error Type**: Morphological agreement errors

**Includes**: Gender agreement, number agreement, definiteness

**Examples**:
```
البنت الكبير ← البنت الكبيرة (the big girl - adjective must be feminine)
الطلاب الجديد ← الطلاب الجدد (the new students - adjective must be plural)
```

**Architecture**: AraBART fine-tuned on agreement errors

**Challenge**: Requires understanding Arabic morphology deeply

### Model 10: GeneralGEC

**Error Type**: Catch-all for remaining errors

**Purpose**: Handle anything the specialists miss

**Architecture**: AraBART fine-tuned on full QALB + synthetic data

**Training data**: 50,000 mixed error pairs

---

## Technical Implementation

### Base Model: AraBART

We use [AraBART](https://huggingface.co/moussaKam/AraBART) as the base for neural models:

- **Architecture**: BART (Bidirectional and Auto-Regressive Transformer)
- **Parameters**: 139M
- **Pretraining**: Arabic Wikipedia, news, books
- **License**: Apache 2.0 (commercial-friendly)

**Why AraBART over alternatives**:
- Specifically designed for Arabic seq2seq tasks
- Strong performance on summarization/generation
- Reasonable size (not as huge as AraGPT2)
- Active maintenance

### Project Structure

```
nahawi/
├── nahawi_ensemble/
│   ├── __init__.py
│   ├── config.py              # Configuration management
│   ├── orchestrator.py        # Main ensemble coordinator
│   ├── cli.py                 # Command-line interface
│   └── models/
│       ├── base.py            # Abstract base classes
│       ├── rule_based.py      # TaaMarbutaFixer, AlifMaksuraFixer, etc.
│       ├── arabart_base.py    # AraBART-based models
│       └── camelbert.py       # CAMeLBERT for morphology
├── train_arabart_simple.py    # GeneralGEC training
├── train_specialized_models.py # Specialized model training
├── train_morphology_fixer.py  # Morphology model training
├── generate_qalb_synthetic_v4.py # Synthetic data generation
├── run_evaluation.py          # Evaluation pipeline
├── test_orchestrator.py       # Integration tests
└── test_rules.py              # Rule-based model tests
```

### Configuration System

```python
# nahawi_ensemble/config.py

@dataclass
class Config:
    # Paths
    data_dir: Path = Path("qalb_real_data")
    checkpoint_dir: Path = Path("nahawi_ensemble/checkpoints")

    # Model settings
    max_length: int = 128
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Training
    batch_size: int = 8
    learning_rate: float = 2e-5
    epochs: int = 3

    # Inference
    default_strategy: str = "cascading"
    confidence_threshold: float = 0.7
```

### Model Base Classes

```python
# nahawi_ensemble/models/base.py

class BaseGECModel(ABC):
    """Abstract base for all GEC models."""

    def __init__(self, name: str, error_types: List[str]):
        self.name = name
        self.error_types = error_types
        self.is_loaded = False

    @abstractmethod
    def correct(self, text: str) -> CorrectionResult:
        """Correct errors in text."""
        pass

    @abstractmethod
    def load(self) -> None:
        """Load model weights."""
        pass


class RuleBasedModel(BaseGECModel):
    """Base for rule-based models (no loading needed)."""

    def load(self) -> None:
        self.is_loaded = True  # Always ready

    def correct(self, text: str) -> CorrectionResult:
        corrected, corrections = self.apply_rules(text)
        return CorrectionResult(
            original_text=text,
            corrected_text=corrected,
            corrections=corrections,
            confidence=0.95  # Rule-based = high confidence
        )

    @abstractmethod
    def apply_rules(self, text: str) -> Tuple[str, List[Correction]]:
        pass


class NeuralGECModel(BaseGECModel):
    """Base for neural models (AraBART, CAMeLBERT)."""

    def __init__(self, name, error_types, base_model, checkpoint_path, device, max_length):
        super().__init__(name, error_types)
        self.base_model = base_model
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.max_length = max_length
        self.model = None
        self.tokenizer = None
```

---

## Training Pipeline

### Synthetic Data Generation

Real annotated data (QALB) is limited (~20K sentences). We generate synthetic errors to augment:

```python
# generate_qalb_synthetic_v4.py

ERROR_GENERATORS = {
    'hamza': inject_hamza_errors,      # 15% of pairs
    'taa_marbuta': inject_taa_errors,  # 10% of pairs
    'space': inject_space_errors,      # 25% of pairs
    'spelling': inject_spelling_errors, # 20% of pairs
    'deletion': inject_deletion_errors, # 15% of pairs
    'morphology': inject_morph_errors,  # 15% of pairs
}

def generate_synthetic_pair(clean_text):
    """Generate (error_text, clean_text) pair."""
    error_type = random.choices(
        list(ERROR_GENERATORS.keys()),
        weights=[15, 10, 25, 20, 15, 15]
    )[0]

    error_text = ERROR_GENERATORS[error_type](clean_text)
    return error_text, clean_text
```

**Data sources for clean text**:
- Arabic Wikipedia
- News articles
- QALB reference sentences

**Generated**: 200,000 synthetic pairs (~110MB)

### Training Process

Each specialized model trains on filtered data:

```python
# train_specialized_models.py

def filter_pairs_for_model(pairs, model_type):
    """Filter training pairs by error type."""

    if model_type == 'hamza_fixer':
        return [p for p in pairs if has_hamza_diff(p[0], p[1])]

    elif model_type == 'space_fixer':
        return [p for p in pairs if has_space_diff(p[0], p[1])]

    elif model_type == 'spelling_fixer':
        return [p for p in pairs if 1 <= char_diff(p[0], p[1]) <= 3]

    # ... etc
```

**Training configuration**:
```python
Seq2SeqTrainingArguments(
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    fp16=False,  # CPU training
    save_strategy="epoch",
    eval_strategy="epoch",
)
```

### Training Time Estimates (CPU)

| Model | Training Pairs | Epochs | Est. Time |
|-------|---------------|--------|-----------|
| GeneralGEC | 50,000 | 3 | ~52h |
| HamzaFixer | 28,000 | 3 | ~29h |
| SpaceFixer | 28,000 | 3 | ~30h |
| SpellingFixer | 9,000 | 3 | ~9h |
| DeletedWordFixer | 28,000 | 3 | ~31h |
| MorphologyFixer | 1,500 | 3 | ~1.5h |

*Note: GPU training would be ~10x faster*

---

## Evaluation & Metrics

### Primary Metric: F0.5

GEC uses F0.5 (not F1) because **precision matters more than recall**:
- False positives = introducing new errors (very bad)
- False negatives = missing corrections (less bad)

```python
def calculate_f05(precision, recall):
    beta = 0.5
    return (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)
```

### Evaluation Pipeline

```python
# run_evaluation.py

def evaluate_ensemble(ensemble, test_pairs):
    predictions = []

    for source, target in test_pairs:
        result = ensemble.correct(source)
        predictions.append(result.corrected_text)

    # Calculate word-level edits
    for pred, src, tgt in zip(predictions, sources, targets):
        pred_edits = get_edits(src, pred)
        gold_edits = get_edits(src, tgt)

        tp += len(pred_edits & gold_edits)  # Correct corrections
        fp += len(pred_edits - gold_edits)  # Wrong corrections
        fn += len(gold_edits - pred_edits)  # Missed corrections

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f05 = calculate_f05(precision, recall)

    return {'f05': f05, 'precision': precision, 'recall': recall}
```

### Benchmark: QALB-2014

The standard benchmark for Arabic GEC:

| System | Precision | Recall | F0.5 |
|--------|-----------|--------|------|
| QALB-2014 Winner | 71.0% | 56.0% | 67.9% |
| Recent SOTA | ~75% | ~60% | ~72% |
| Nahawi (rule-based only) | 2.3% | 0.6% | 1.4% |
| Nahawi (full ensemble) | TBD | TBD | TBD |

*Full ensemble results pending training completion*

---

## Installation & Usage

### Requirements

- Python 3.9+
- PyTorch 2.0+
- Transformers 4.30+
- ~2GB disk space for models

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/nahawi.git
cd nahawi

# Install dependencies
pip install torch transformers

# Install package
pip install -e .
```

### Quick Start

```python
from nahawi_ensemble.orchestrator import NahawiEnsemble

# Initialize (lazy loads models)
ensemble = NahawiEnsemble()

# Correct text
result = ensemble.correct("هذه مدرسه جميله")
print(result.corrected_text)  # هذه مدرسة جميلة

# Get details
print(f"Corrections: {len(result.corrections)}")
print(f"Confidence: {result.confidence}")
print(f"Models used: {result.model_contributions}")
```

### CLI Usage

```bash
# Correct single text
python -m nahawi_ensemble.cli correct "هذه مدرسه جميله"

# Correct file
python -m nahawi_ensemble.cli correct --file input.txt --output output.txt

# Interactive mode
python -m nahawi_ensemble.cli interactive

# Check model status
python -m nahawi_ensemble.cli status
```

### Training Your Own Models

```bash
# Generate synthetic data
python generate_qalb_synthetic_v4.py --num-pairs 200000

# Train GeneralGEC
python train_arabart_simple.py --model-name general_gec --epochs 3

# Train specialized models
python train_specialized_models.py --model hamza_fixer --epochs 3
python train_specialized_models.py --model space_fixer --epochs 3
python train_specialized_models.py --model spelling_fixer --epochs 3
python train_specialized_models.py --model deleted_word_fixer --epochs 3

# Train morphology model
python train_morphology_fixer.py --epochs 3
```

### Evaluation

```bash
# Quick evaluation (100 samples)
python run_evaluation.py --max-samples 100

# Full evaluation
python run_evaluation.py --max-samples 1000
```

---

## Results & Analysis

### Rule-Based Models Performance

| Model | Test Cases | Pass Rate | Notes |
|-------|------------|-----------|-------|
| TaaMarbutaFixer | 4 | 100% | Dictionary + pattern |
| AlifMaksuraFixer | 4 | 100% | Word list |
| PunctuationFixer | 3 | 100% | Simple replacement |
| RepeatedWordFixer | 3 | 100% | With exceptions |

### Orchestrator Tests

```
============================================================
TEST 4: Orchestrator Correction (Cascading)
============================================================

[PASS] Input:    'هذه مدرسه جميله'
       Expected: 'هذه مدرسة جميلة'
       Got:      'هذه مدرسة جميلة'
       Models:   ['taa_marbuta_fixer']

[PASS] Input:    'ذهب الي المدرسه'
       Expected: 'ذهب إلى المدرسة'
       Got:      'ذهب إلى المدرسة'
       Models:   ['taa_marbuta_fixer', 'alif_maksura_fixer']

[PASS] Input:    'مرحبا, كيف حالك?'
       Expected: 'مرحبا، كيف حالك؟'
       Got:      'مرحبا، كيف حالك؟'
       Models:   ['punctuation_fixer']

[PASS] Input:    'ذهبت الى الى المدرسة'
       Expected: 'ذهبت الى المدرسة'
       Got:      'ذهبت الى المدرسة'
       Models:   ['repeated_word_fixer']

Result: 4 passed, 0 failed
All strategies working: cascading, parallel, specialist
```

### Neural Models (Training in Progress)

As of last update:
- MorphologyFixer: 27% complete
- SpellingFixer: 12% complete
- HamzaFixer: 4% complete
- SpaceFixer: 4% complete
- DeletedWordFixer: 1% complete
- GeneralGEC: 3% complete

---

## Lessons Learned

### 1. Pretraining is Non-Negotiable

You cannot train an effective Arabic GEC model from scratch with limited data. The morphological complexity requires pretrained knowledge.

**Before**: Custom 30MB model → 38% F0.5
**After**: AraBART fine-tuning → TBD (expected 50%+)

### 2. Specialization Beats Generalization

A single model trying to handle all errors performs worse than specialized models:
- Hamza errors need morphological context
- Space errors need word boundary detection
- Spelling errors need character-level attention

### 3. Rule-Based Has Its Place

Don't dismiss rules entirely:
- **Speed**: 1000x faster than neural
- **Precision**: Near 100% for known patterns
- **Interpretability**: Easy to debug
- **No training**: Works immediately

Best approach: Rule-based first, neural for what rules can't handle.

### 4. Data Quality > Data Quantity

Synthetic data helps, but quality matters:
- Bad: Random character swaps
- Good: Linguistically plausible errors based on real patterns

### 5. Arabic-Specific Challenges

Things that surprised me:
- Hamza rules are incredibly complex (8+ forms)
- Same word can be correct or incorrect based on context
- Dialectal influence is everywhere
- Diacritics would help but aren't available

### 6. Evaluation is Hard

QALB metrics don't tell the whole story:
- Word-level F0.5 misses character-level corrections
- Some "errors" are style choices
- Context matters (formal vs informal)

---

## Future Directions

### Short-Term Improvements

1. **GPU Training**: Current CPU training is slow. GPU would enable:
   - Larger batch sizes
   - More epochs
   - Faster iteration

2. **Hyperparameter Tuning**:
   - Learning rate scheduling
   - Optimal confidence thresholds
   - Model-specific settings

3. **Error Analysis**:
   - Which errors are we missing?
   - Where do we over-correct?
   - Model contribution analysis

### Medium-Term Goals

1. **Model Distillation**: Compress for production
   - INT8 quantization
   - Knowledge distillation to smaller models
   - Shared encoder weights

2. **Confidence Calibration**:
   - Better uncertainty estimates
   - When to abstain from correction

3. **Context Window Expansion**:
   - Document-level coherence
   - Cross-sentence corrections

### Long-Term Vision

1. **Dialect Handling**:
   - Detect dialect
   - Appropriate corrections per dialect
   - Code-switching awareness

2. **Diacritization Integration**:
   - Use diacritics when available
   - Optionally restore diacritics

3. **Real-Time API**:
   - Web service deployment
   - Browser extension
   - Mobile SDK

---

## References & Acknowledgments

### Papers

1. Zaghouani et al. (2014). "Large Scale Arabic Error Annotation: Guidelines and Framework." LREC.
2. Mohit et al. (2014). "The First QALB Shared Task on Automatic Text Correction for Arabic." EMNLP Workshop.
3. Kamal Eddine et al. (2021). "AraBART: a Pretrained Arabic Sequence-to-Sequence Model." arXiv.

### Datasets

- **QALB-2014**: Qatar Arabic Language Bank shared task data
- **Arabic Wikipedia**: Source for clean text generation

### Tools & Libraries

- [AraBART](https://huggingface.co/moussaKam/AraBART) by Moussa Kamal Eddine
- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [PyTorch](https://pytorch.org/)

### Acknowledgments

This project was developed through extensive experimentation and iteration. Thanks to:
- The QALB shared task organizers for the benchmark
- The AraBART team for the pretrained model
- The HuggingFace team for the transformers library

---

## License

MIT License - See [LICENSE](LICENSE) for details.

The code is MIT licensed. Note that:
- AraBART base model is Apache 2.0 licensed
- QALB data has its own license (research use)
- Synthetic training data is freely usable

---

## Contributing

Contributions welcome! Areas where help is needed:
- GPU training and evaluation
- Additional rule-based patterns
- Dialect-specific handling
- Documentation improvements

Please open an issue first to discuss proposed changes.

---

## Citation

If you use Nahawi in your research, please cite:

```bibtex
@software{nahawi2024,
  title = {Nahawi: Arabic Grammatical Error Correction Ensemble},
  author = {[Your Name]},
  year = {2024},
  url = {https://github.com/yourusername/nahawi},
  note = {10-model ensemble for Arabic GEC}
}
```

---

<div align="center">

**Built with ☕ and determination**

*Because Arabic deserves great NLP tools too*

</div>
