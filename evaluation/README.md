# Evaluation Scripts

Scripts for evaluating the Nahawi GEC system on FASIH v4.1 benchmark.

## Files

### eval_hybrid.py
Main evaluation script for the V5 + Rules hybrid system.

**Usage:**
```bash
python eval_hybrid.py
```

**Output:**
```
============================================================
V5 + RULES RESULTS
============================================================
alif_maqsura                  : F0.5= 97.4% [PASS]
definiteness                  : F0.5= 99.3% [PASS]
gender_agreement              : F0.5= 92.3% [PASS]
...
============================================================
OVERALL F0.5: 94.7%
PASS categories: 11/13
============================================================
```

### rule_postprocessor.py
Standalone rule-based post-processor for boosting neural model output.

**Rules implemented:**
1. **Alif Maqsura**: الي → إلى
2. **Gender Agreement**: Masculine noun + feminine adjective → masculine adjective
3. **Taa Marbuta**: Common ه → ة fixes
4. **Preposition Fixes**: يعتمد عن → يعتمد على

**Usage:**
```python
from rule_postprocessor import RulePostProcessor

pp = RulePostProcessor()
corrected = pp.process("قمر صغيرة")
print(corrected)  # قمر صغير
```

## Metrics

### F0.5
The primary metric for GEC evaluation. Weighs precision more than recall:
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- F0.5 = (1.25 × P × R) / (0.25 × P + R)

### Pass Thresholds
- **PASS**: F0.5 >= 90%
- **WORK**: F0.5 >= 80%
- **FAIL**: F0.5 < 80%

## Requirements

```
torch
sentencepiece
```

## Configuration

The neural model configuration:
```python
CONFIG = {
    'vocab_size': 32000,
    'd_model': 768,
    'nhead': 12,
    'num_encoder_layers': 6,
    'num_decoder_layers': 6,
    'dim_feedforward': 3072,
    'dropout': 0.1,
    'max_seq_len': 256
}
```

## Adding New Rules

To add new rules to the post-processor:

1. Identify patterns the neural model consistently misses
2. Add to appropriate dictionary in `RulePostProcessor.__init__`
3. Test for false positives (rules that incorrectly change correct text)
4. Run full evaluation to confirm improvement

Example:
```python
# In RulePostProcessor.__init__:
self.new_fixes = {
    'error_pattern': 'correct_pattern',
}

# In process():
for wrong, correct in self.new_fixes.items():
    result = result.replace(wrong, correct)
```
