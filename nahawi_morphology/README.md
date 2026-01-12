# Nahawi Arabic Morphology Analyzer

Elite-grade Arabic morphological analysis and processing toolkit.

## Features

- **Word Analysis**: Extract root, pattern, gender, number, POS
- **Agreement Checking**: Subject-verb, noun-adjective agreement validation
- **Word Generation**: Generate words from roots and patterns
- **Verb Conjugation**: Full conjugation paradigms
- **Error Injection**: Generate training data with realistic errors
- **Post-processing**: Fix agreement errors in text

## Quick Start

```python
from nahawi_morphology import ArabicMorphology

morph = ArabicMorphology()

# Analyze a word
info = morph.analyze("الطالبات")
print(f"Gender: {info.gender}")   # feminine
print(f"Number: {info.number}")   # plural
print(f"Definite: {info.definite}")  # True

# Check agreement
errors = morph.check_agreement("الطالبة", "المجتهد")
# Returns: [AgreementError(type='gender', word1='الطالبة', word2='المجتهد')]

# Fix agreement in sentence
fixed = morph.fix_agreement("الطالبة المجتهد")
print(fixed)  # الطالبة المجتهدة

# Generate word
word = morph.generate(root='كتب', pattern='فاعل', gender='fem')
print(word)  # كاتبة

# Conjugate verb
verb = morph.conjugate(root='ذهب', tense='past', person='3fs')
print(verb)  # ذهبت

# Inject error for training data
error_word = morph.inject_error("ذهبت", "gender")
print(error_word)  # ذهب (masculine instead of feminine)
```

## API Reference

### ArabicMorphology

Main interface class combining all morphology tools.

#### analyze(word: str) -> MorphInfo
Analyze a word and return morphological information.

```python
info = morph.analyze("المدرسة")
# MorphInfo(root='درس', pattern='مفعلة', gender='fem', number='sing',
#           pos='noun', definite=True)
```

#### check_agreement(word1: str, word2: str) -> list
Check agreement between two words.

```python
errors = morph.check_agreement("البنت", "الذكي")
# [AgreementError(type='gender', ...)]
```

#### fix_agreement(sentence: str) -> str
Fix agreement errors in a sentence.

```python
fixed = morph.fix_agreement("البنت الذكي")
# "البنت الذكية"
```

#### conjugate(root: str, tense: str, person: str, form: int) -> str
Conjugate a verb from its root.

```python
morph.conjugate("كتب", "past", "1s")  # كتبت
morph.conjugate("كتب", "present", "3ms")  # يكتب
morph.conjugate("كتب", "imperative", "2ms")  # اكتب
```

#### generate(root: str, pattern: str, gender: str, number: str, definite: bool) -> str
Generate a word from root and pattern.

```python
morph.generate("كتب", "فاعل", "fem", "plural")  # كاتبات
morph.generate("علم", "مفعول", "masc", "sing", True)  # المعلوم
```

#### inject_error(word: str, error_type: str) -> str
Inject a specific type of error into a word.

Error types: `gender`, `number`, `taa_marbuta`, `hamza`, `alif_maqsura`

```python
morph.inject_error("كتبت", "gender")  # كتب
morph.inject_error("كاتبة", "taa_marbuta")  # كاتبه
```

### MorphInfo

Dataclass returned by analyze():

```python
@dataclass
class MorphInfo:
    word: str           # Original word
    root: str           # Triliteral root
    pattern: str        # Morphological pattern
    gender: str         # 'masc' or 'fem'
    number: str         # 'sing', 'dual', or 'plural'
    pos: str            # Part of speech
    definite: bool      # Has definite article
    case: str           # 'nom', 'acc', 'gen' (if detectable)
```

## Files

| File | Description |
|------|-------------|
| `__init__.py` | Package exports and ArabicMorphology class |
| `analyzer.py` | Word analysis (root extraction, feature detection) |
| `agreement.py` | Agreement checking and fixing |
| `conjugator.py` | Verb conjugation |
| `generator.py` | Word generation from roots |
| `error_injector.py` | Error injection for training data |
| `postprocessor.py` | Rule-based post-processing |
| `data.py` | Morphological data (patterns, lexicons) |

## Use Cases

### GEC Model Evaluation

```python
# Check if model output has correct agreement
output = model.correct("الولد الجميلة")
errors = morph.check_agreement(*output.split()[-2:])
if errors:
    print("Model failed to fix agreement")
```

### Training Data Generation

```python
# Generate erroneous versions of correct text
correct = "ذهبت الطالبة إلى المدرسة"
words = correct.split()
for i, word in enumerate(words):
    error_word, error_type = morph.inject_random_error(word)
    if error_word != word:
        erroneous = words[:i] + [error_word] + words[i+1:]
        training_pair = {
            "source": " ".join(erroneous),
            "target": correct,
            "error_type": error_type
        }
```

### Post-processing

```python
# Apply morphology-based fixes after neural model
neural_output = model.correct(text)
final_output = morph.postprocess(neural_output)
```

## Tests

```bash
python -m pytest nahawi_morphology/tests/
```

## See Also

- `arabic_conjugator/` - Standalone conjugation library
- `roots/` - Root database
- `evaluation/rule_postprocessor.py` - Rule-based post-processor
