# Nahawi Model Weights

Model weights are not included in the repository (too large for git).

## Model Hierarchy

```
nahawi_base.pt (474MB)          # Pure pretrained LM on 23M MSA sentences
       ↓ GEC Fine-tuning
fasih_v15_model.pt (1.4GB)      # GEC-tuned, 71.82% F0.5 (with punct)
       ↓ Hamza LoRA
(hamza-focused training)        # Improved hamza correction ~74% F0.5
       ↓ Punct-aware LoRA
punct_aware_lora/*.pt (30MB)    # +7.0 points → 78.84% F0.5 (with punct)
```

**Training chain:** V15 base → Hamza LoRA → Punct-aware LoRA (epoch 3 is best)

## Directory Structure

```
models/
├── base/
│   ├── nahawi_base.pt           # 474MB - Pure pretrained (23M MSA sentences)
│   ├── nahawi_base_config.json  # Pretraining config
│   └── fasih_v15_model.pt       # 1.4GB - GEC fine-tuned (use this as LoRA base)
├── punct_aware_lora/
│   ├── best_lora.pt             # 30MB - Best LoRA checkpoint
│   ├── epoch_1.pt               # 30MB - Epoch 1 checkpoint
│   ├── epoch_2.pt               # 30MB - Epoch 2 checkpoint
│   └── epoch_3.pt               # 30MB - BEST (78.84% F0.5)
└── tokenizer/
    └── (see project root: nahawi_spm.model)
```

Note: The tokenizer is in the project root as `nahawi_spm.model` (900KB).

## Model Descriptions

### nahawi_base.pt (Pure Pretrained)
- **Size**: 474MB (124M parameters)
- **Training**: Unsupervised LM on 23M MSA sentences
- **Corpus**: Wikipedia + UN + News + Shamela (6.2GB)
- **Use case**: Base for new fine-tuning experiments
- **Vocab**: 32K SentencePiece

### fasih_v15_model.pt (GEC Fine-tuned)
- **Size**: 1.4GB (124M parameters + optimizer state)
- **Training**: Supervised GEC on QALB + synthetic data
- **Performance**: 71.82% F0.5 (with punct), 87.63% (no punct)
- **Use case**: Base model for LoRA adapters (use this for inference)

### punct_aware_lora/*.pt (LoRA Adapters)
- **Size**: ~30MB each
- **Training**: Punct-aware LoRA (rank=64, alpha=128), built on top of Hamza LoRA
- **Chain**: V15 base → Hamza LoRA → Punct-aware LoRA
- **Key insight**: QALB (86.5%) teaches punctuation, stripped synthetic teaches content
- **Performance**: 78.84% F0.5 with punctuation (epoch 3 is best)

## Download from Remote Server

```bash
# Base model (1.4GB) - GEC fine-tuned (USE THIS)
scp -i nahawi.pem ubuntu@192.222.50.72:/home/ubuntu/nahawi/fasih_v15_model/best_model.pt ./models/base/fasih_v15_model.pt

# Pure pretrained (474MB) - for new experiments
scp -i nahawi.pem ubuntu@192.222.50.72:/home/ubuntu/nahawi/pretrained_nahawi/nahawi_base.pt ./models/base/

# LoRA weights (30MB each) - epoch_3 is BEST
scp -i nahawi.pem ubuntu@192.222.50.72:/home/ubuntu/nahawi/lora_punct_aware/epoch_3.pt ./models/punct_aware_lora/
scp -i nahawi.pem ubuntu@192.222.50.72:/home/ubuntu/nahawi/lora_punct_aware/epoch_1.pt ./models/punct_aware_lora/
scp -i nahawi.pem ubuntu@192.222.50.72:/home/ubuntu/nahawi/lora_punct_aware/epoch_2.pt ./models/punct_aware_lora/
scp -i nahawi.pem ubuntu@192.222.50.72:/home/ubuntu/nahawi/lora_punct_aware/best_lora.pt ./models/punct_aware_lora/
```

## Model Performance

| Model | F0.5 (with punct) | F0.5 (no punct) | Notes |
|-------|-------------------|-----------------|-------|
| Base (V15) | 71.82% | 87.63% | Good content, weak punct |
| + Hamza LoRA | ~74% | ~89% | Hamza-focused fine-tuning |
| + Punct-aware Epoch 1 | 78.11% | 91.47% | +6.3 points from V15 |
| + Punct-aware Epoch 2 | 78.52% | 91.95% | Good |
| **+ Punct-aware Epoch 3** | **78.84%** | **91.89%** | **Best checkpoint** |
| + Punct-aware Epoch 4 | 77.96% | 91.87% | Regression |
| **SOTA (ArbESC+)** | **82.63%** | ??? | 3.79 points ahead |

## Model Architecture

### Base Model (124M parameters)
```
vocab_size: 32,000 (SentencePiece)
d_model: 768
nhead: 12
num_encoder_layers: 6
num_decoder_layers: 6
dim_feedforward: 3072
dropout: 0.1
max_seq_len: 256
```

### LoRA Config
```
rank: 64
alpha: 128
dropout: 0.05 (training) / 0.0 (inference)
target_layers: attention out_proj, FFN linear1/linear2
```

## Loading Models

### Python API
```python
from nahawi import NahawiModel

model = NahawiModel()
model.load()  # Auto-loads base + LoRA

corrected, corrections = model.correct("اعلنت الحكومه عن خطه جديده")
print(corrected)  # أعلنت الحكومة عن خطة جديدة
```

### Custom Paths
```python
model = NahawiModel(
    model_path="path/to/fasih_v15_model.pt",
    lora_path="path/to/best_lora.pt",
    spm_path="path/to/nahawi_spm.model"
)
```

## Training Your Own

### LoRA Fine-tuning
```bash
# Prepare punct-aware data
python scripts/prepare_punct_aware_data.py

# Train LoRA
python scripts/train_lora_punct_aware.py
```

### New Base Model Pretraining
Requires the MSA corpus (`corpus/msa_corpus_full.txt`, 6.2GB).

See `lora_experiment/` for training scripts and configuration.

## File Checksums

```
fasih_v15_model.pt  MD5: [compute with: certutil -hashfile models/base/fasih_v15_model.pt MD5]
nahawi_base.pt      MD5: [compute with: certutil -hashfile models/base/nahawi_base.pt MD5]
best_lora.pt        MD5: [compute with: certutil -hashfile models/punct_aware_lora/best_lora.pt MD5]
```
