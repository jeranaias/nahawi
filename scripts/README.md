# Nahawi Training Scripts

Production training scripts for Nahawi GEC models.

## Scripts

### prepare_punct_aware_data.py

Prepares the punct-aware training dataset for LoRA fine-tuning.

**Key insight**: QALB data (86.5% of training) teaches punctuation, while synthetic/hamza data has punctuation stripped so the model learns content correction without learning wrong punct patterns.

```bash
python scripts/prepare_punct_aware_data.py
```

**Input files required**:
- `data/qalb_real_train.json` - 36,771 QALB pairs (with punctuation)
- `data/hamza_augmentation.json` - Hamza error augmentation (punct stripped)
- Synthetic data (punct stripped)

**Output**:
- `data/punct_aware/train_punct_aware.json` - Combined training data

**Data composition**:
| Source | Count | Ratio | Purpose |
|--------|-------|-------|---------|
| QALB x50 | 1,838,550 | 86.5% | Teaches punctuation |
| Synthetic | 150,000 | 7.1% | Teaches content errors |
| Hamza aug | 137,972 | 6.5% | Teaches hamza patterns |
| **Total** | 2,126,522 | 100% | |

### train_lora_punct_aware.py

Trains LoRA adapters on the punct-aware dataset.

```bash
python scripts/train_lora_punct_aware.py
```

**Configuration**:
```python
lora_rank: 64
lora_alpha: 128
lora_dropout: 0.05
batch_size: 224
learning_rate: 3e-4
num_epochs: 3
```

**Results**:
| Epoch | F0.5 (with punct) | F0.5 (no punct) |
|-------|-------------------|-----------------|
| 1 | 78.11% | 91.47% |
| 2 | 78.52% | 91.95% |

## Usage Pattern

1. **Prepare data** (run once):
   ```bash
   python scripts/prepare_punct_aware_data.py
   ```

2. **Train LoRA** (run on GPU):
   ```bash
   python scripts/train_lora_punct_aware.py
   ```

3. **Evaluate**:
   ```bash
   python lora_experiment/eval_lora.py
   ```

## Requirements

- PyTorch with CUDA
- sentencepiece
- tqdm
- ~16GB VRAM minimum (batch_size=224 needs ~90GB for full speed)

## Output

Checkpoints saved to `models/punct_aware_lora/`:
- `epoch_1.pt` - Epoch 1 checkpoint
- `epoch_2.pt` - Epoch 2 checkpoint
- `best_lora.pt` - Best validation loss

## See Also

- `lora_experiment/` - Development scripts and experiments
- `models/README.md` - Model weights documentation
- `data/README.md` - Training data documentation
