# Arabic Verb Conjugator

A comprehensive Arabic verb conjugation engine that handles all root types and verb forms.

## Features

- All 10 verb forms (I-X)
- All root types (sound, hollow, defective, hamzated, doubled, etc.)
- All tenses (past, present, imperative)
- All persons, numbers, genders
- All moods (indicative, subjunctive, jussive)
- Active and passive voice

## Usage

```python
from arabic_conjugator import ArabicConjugator, VerbForm, Tense

conjugator = ArabicConjugator()

# Conjugate a sound root (كتب - write)
forms = conjugator.conjugate_root("كتب", VerbForm.I)
for form in forms:
    print(f"{form.label}: {form.form}")

# Specific tense only
past_forms = conjugator.conjugate_root(
    "كتب",
    VerbForm.I,
    tenses=[Tense.PAST]
)

# Include passive voice
all_forms = conjugator.conjugate_root(
    "كتب",
    VerbForm.I,
    include_passive=True
)
```

## Verb Forms

| Form | Pattern | Example Root | Meaning |
|------|---------|--------------|---------|
| I | فَعَلَ | كتب | basic (wrote) |
| II | فَعَّلَ | كتب | intensive (taught to write) |
| III | فَاعَلَ | كتب | reciprocal (corresponded) |
| IV | أَفْعَلَ | كتب | causative (dictated) |
| V | تَفَعَّلَ | كتب | reflexive of II |
| VI | تَفَاعَلَ | كتب | reciprocal of III |
| VII | اِنْفَعَلَ | كسر | passive-reflexive (was broken) |
| VIII | اِفْتَعَلَ | كتب | reflexive (subscribed) |
| IX | اِفْعَلَّ | حمر | colors/defects (turned red) |
| X | اِسْتَفْعَلَ | كتب | requestative (asked to write) |

## Root Types

The conjugator automatically detects and handles:

| Type | Pattern | Example | Notes |
|------|---------|---------|-------|
| Sound | Regular | كتب | No weak letters |
| Hollow | ع/و or ع/ي | قول, بيع | Middle weak letter |
| Defective | ل/و or ل/ي | دعو, بني | Final weak letter |
| Assimilated | ف/و or ف/ي | وصل, يسر | Initial weak letter |
| Hamzated | Has ء | سأل, قرأ | Contains hamza |
| Doubled | ل = ع | مدد, ردد | 2nd = 3rd radical |

## Output Format

Each conjugated form returns a `ConjugatedForm` object:

```python
@dataclass
class ConjugatedForm:
    form: str           # The conjugated word (e.g., "كَتَبَ")
    root: str           # The root (e.g., "كتب")
    verb_form: VerbForm # Form I-X
    tense: Tense        # PAST, PRESENT, IMPERATIVE
    person: Person      # FIRST, SECOND, THIRD
    number: Number      # SINGULAR, DUAL, PLURAL
    gender: Gender      # MASCULINE, FEMININE
    mood: Mood          # INDICATIVE, SUBJUNCTIVE, JUSSIVE (present only)
    voice: Voice        # ACTIVE, PASSIVE
    label: str          # Human-readable label (e.g., "3ms.past")
```

## Files

| File | Description |
|------|-------------|
| `__init__.py` | Package exports |
| `conjugator.py` | Main ArabicConjugator class |
| `root_types.py` | Root classification logic |
| `verb_forms.py` | Verb form patterns (I-X) |
| `paradigms.py` | Conjugation paradigms |

## Use Cases

### Training Data Generation
Generate correct verb forms for training GEC models:

```python
from arabic_conjugator import ArabicConjugator

conjugator = ArabicConjugator()
roots = ["كتب", "قرأ", "علم", "درس"]

for root in roots:
    forms = conjugator.conjugate_root(root)
    for form in forms:
        # Use form.form as correct target
        pass
```

### Grammar Validation
Check if a verb form is correctly conjugated:

```python
# Generate all valid forms for a root
valid_forms = {f.form for f in conjugator.conjugate_root("كتب")}

# Check if user input is valid
if user_verb in valid_forms:
    print("Correct!")
else:
    print("Error in conjugation")
```

## Limitations

- Focuses on triliteral (3-letter) roots
- Quadriliteral roots not yet supported
- Some rare irregular verbs may need special handling

## See Also

- `nahawi/model.py` - GEC model that uses morphological patterns
- `roots/` - Root dictionary data
