# Nahawi Python Package

The core Nahawi GEC model as a Python package.

## Installation

```bash
pip install torch sentencepiece
```

## Usage

```python
from nahawi import NahawiModel

# Initialize and load model
model = NahawiModel()
model.load()

# Correct Arabic text
text = "اعلنت الحكومه عن خطه جديده"
corrected, corrections = model.correct(text)

print(f"Original: {text}")
print(f"Corrected: {corrected}")
print(f"Corrections: {len(corrections)}")

for c in corrections:
    print(f"  - {c['original']} → {c['corrected']} ({c['error_type']})")
```

## Output

```
Original: اعلنت الحكومه عن خطه جديده
Corrected: أعلنت الحكومة عن خطة جديدة
Corrections: 4
  - اعلنت → أعلنت (hamza)
  - الحكومه → الحكومة (taa_marbuta)
  - خطه → خطة (taa_marbuta)
  - جديده → جديدة (taa_marbuta)
```

## API Reference

### NahawiModel

```python
class NahawiModel:
    def __init__(
        self,
        model_path=None,    # Path to base model (default: models/base/fasih_v15_model.pt)
        lora_path=None,     # Path to LoRA weights (default: models/punct_aware_lora/best_lora.pt)
        spm_path=None       # Path to tokenizer (default: nahawi_spm.model)
    )

    def load(self) -> None:
        """Load model weights and tokenizer."""

    def correct(self, text: str) -> Tuple[str, List[dict]]:
        """
        Correct Arabic text.

        Returns:
            Tuple of (corrected_text, corrections)

        Each correction is a dict:
            - original: Original word
            - corrected: Corrected word
            - start: Start position in corrected text
            - end: End position in corrected text
            - error_type: Type of error (hamza, taa_marbuta, etc.)
            - confidence: Correction confidence (0-1)
        """
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| NAHAWI_MODEL_PATH | Path to base model | models/base/fasih_v15_model.pt |
| NAHAWI_LORA_PATH | Path to LoRA weights | models/punct_aware_lora/best_lora.pt |
| NAHAWI_SPM_PATH | Path to tokenizer | nahawi_spm.model |

## Error Types

| Type | Description | Example |
|------|-------------|---------|
| hamza | Hamza placement | اعلنت → أعلنت |
| taa_marbuta | Taa marbuta vs ha | الحكومه → الحكومة |
| alif_maqsura | Alif maqsura vs ya | الي → إلى |
| letter_confusion | Similar letters | دهب → ذهب |
| punctuation | Punctuation errors | |
| spelling | Other spelling | |

## Requirements

- Python 3.9+
- PyTorch
- sentencepiece

## Files

| File | Description |
|------|-------------|
| `__init__.py` | Package exports |
| `model.py` | NahawiModel class with LoRA support |
