# Nahawi Project Context

## What Is Nahawi?
Arabic Grammatical Error Correction (GEC) system - "Arabic Grammarly" - targeting **85%+ F0.5** to beat SOTA (ArbESC+ at 82.63%).

---

## CURRENT BEST RESULT (January 11, 2025)

### Punct-Aware LoRA Epoch 3: **78.84% F0.5**

| Metric | Score | Gap to SOTA |
|--------|-------|-------------|
| **F0.5 WITH Punct** | **78.84%** | **3.79 points** |
| F0.5 NO Punct | 91.89% | N/A |
| FASIH F0.5 | 85.93% | N/A |

**Best Checkpoint:** `models/punct_aware_lora/epoch_3.pt` (on V15 base)

### What Worked
- Punct-aware training data (QALB×50 with punct + synthetic without punct)
- LoRA fine-tuning on V15 base model
- 3 epochs optimal (epoch 4 regresses)

### What Didn't Work
- **Two-pass correction:** -2.04% worse than single-pass
- **Continued training:** Epoch 4 = 77.96% (worse than epoch 3)

---

## MODELS

| Model | Location | F0.5 (with punct) | Notes |
|-------|----------|-------------------|-------|
| **punct_aware_lora/epoch_3.pt** | `models/` | **78.84%** | BEST - use this |
| fasih_v15_model/best_model.pt | server | ~55% | Base for LoRA |
| gec_clean/epoch_11.pt | server | 55.32% | Old best content |

---

## KEY FINDINGS

### Two-Pass is a Dead End
| Benchmark | Single Pass | Two Pass | Diff |
|-----------|-------------|----------|------|
| QALB | 78.83% | 76.79% | **-2.04%** |
| FASIH | 85.93% | 85.73% | **-0.20%** |

### Epoch Progression
| Epoch | F0.5 WITH Punct | F0.5 NO Punct |
|-------|-----------------|---------------|
| 3 | **78.84%** | 91.89% |
| 4 | 77.96% | 91.87% |

Epoch 3 is optimal. Further training causes regression.

---

## EXPERT ANALYSIS & RECOMMENDATIONS

### Key Insights (from external expert review)

1. **1:1800 real-to-synthetic ratio is dangerous**
   - 37K QALB vs 66M synthetic teaches the model our error generation function, not real Arabic errors
   - Explains strong hamza/taa performance (easy to synthesize) vs weak complex edits

2. **Punct is a different task masquerading as the same task**
   - 81% no-punct → 55% with-punct = punct is the problem
   - Binary classifier probably insufficient for 27-point gap
   - Options: Multi-class classifier → BIO+CRF → Separate seq2seq

3. **MoE + Copy mechanism is high-risk**
   - V16 collapsed due to copy mechanism breaking
   - MoE adds routing instability
   - Monitor copy gate statistics during training

4. **30% identity might not be enough**
   - If forgetting happened at epoch 1, may need 50%
   - Consider explicit copy gate supervision

### Recommended Approach: LoRA Experiment

**Hypothesis being tested:**
```
IF LoRA on 124M + better data balance beats 82.63%:
   → Data distribution was the bottleneck, not model capacity
   → Kill the 700M pretrain, iterate on data

IF LoRA plateaus at ~55% (current with-punct ceiling):
   → Capacity might be the constraint
   → Continue 700M pretrain with architectural fixes
```

**The Experiment:**
1. Use epoch_11 as base (best content at 81%)
2. LoRA fine-tune with rank 32 on attention + FFN
3. Data: 1:4 ratio (QALB repeated 4x + 150K synthetic)
4. Train separate punct classifier on QALB + 5.9GB MSA corpus
5. Evaluate combined system

**WHY DIFFERENT DATA FOR EACH:**
```
LoRA (error correction):
  - Needs parallel error data: incorrect → correct pairs
  - Uses QALB + synthetic (real error distribution)
  - MSA can't help here - it's clean, no error pairs

Punct Classifier:
  - Learning "where does punct naturally occur in well-formed Arabic"
  - Uses 5.9GB MSA corpus directly (correct punct patterns)
  - No corruption needed - clean text is the training signal
```

**Expected Outcomes:**
| Result | Interpretation |
|--------|----------------|
| LoRA + punct > 75% | Major win - data + task decomposition works |
| LoRA + punct 65-75% | Promising - punct classifier needs work |
| LoRA + punct 55-65% | Punct classifier failed - try seq2seq |
| LoRA + punct < 55% | Something broke - check content regression |

---

## DATA INVENTORY

### Real Data
```
/home/ubuntu/nahawi/data/qalb_real_train.json  - 36,771 pairs (GOLD)
/home/ubuntu/nahawi/data/qalb_real_dev.json    - Dev set
```

### MSA Corpus (5.9GB - ELITE for punct training)
```
/home/ubuntu/nahawi/corpus/combined/msa_corpus_full.txt  - 5.9GB combined
/home/ubuntu/nahawi/corpus/wikipedia/wiki_clean.txt      - 1.6GB
/home/ubuntu/nahawi/corpus/un/un_clean.txt               - 3.5GB
/home/ubuntu/nahawi/corpus/news/leipzig_clean.txt        - 511MB
/home/ubuntu/nahawi/corpus/shamela/shamela_clean.txt     - 440MB
```

### Synthetic Data (66M deduplicated)
```
/home/ubuntu/nahawi/data/ultimate/
├── train_full.json      - 66M pairs (19GB)
├── train_easy.json      - 46M pairs (99.75%)
├── train_medium.json    - 115K pairs
├── train_hard.json      - 563 pairs
└── train_identity.json  - 20M pairs
```

### V9 Bridge Data (used for V15)
```
/home/ubuntu/nahawi/data/v9_bridge/  - 256K pairs
├── fasih_style: 73K
├── long_msa: 100K
├── punct_training: 50K
└── qalb_real: 34K (13%)
```

---

## LORA EXPERIMENT SCRIPTS

Located in `c:\nahawi\lora_experiment\`:

| Script | Purpose |
|--------|---------|
| `sample_balanced_300k.py` | Create 1:4 ratio dataset (QALB 4x + 150K synthetic) |
| `train_lora_v2.py` | LoRA fine-tuning (rank 32, attn+FFN) on epoch_11 |
| `train_punct_classifier_v2.py` | Punct classifier trained on QALB + 5.9GB MSA |
| `eval_lora.py` | Evaluation with F0.5 (with + without punct) |

### LoRA Config
```python
base_model: epoch_11.pt (81% no-punct)
lora_rank: 32
lora_alpha: 64
target_layers: attention out_proj + FFN linear1/linear2
learning_rate: 2e-4
epochs: 3
data: 300K (150K QALB repeated + 150K synthetic stratified)
```

### Punct Classifier Config
```python
base_encoder: epoch_11.pt (frozen)
head: Linear(768→768) + LayerNorm + GELU + Linear(768→384) + GELU + Linear(384→num_punct)
training_data: QALB targets + 5.9GB MSA corpus (500K samples/epoch)
punct_classes: ، ؛ ؟ ! . , : ; ? (9 + no_punct = 10)
```

---

## EVALUATION METHODOLOGY

**Always evaluate WITH punct** - that's what SOTA uses.

Report three metrics:
1. **F0.5 with punct** (PRIMARY - compare to SOTA 82.63%)
2. **F0.5 no punct** (diagnostic - shows content capability)
3. **Punct-only accuracy** (diagnostic - shows punct model quality)

---

## SERVER STATE

```bash
ssh -i "C:\Users\Jesse\Downloads\nahawi.pem" ubuntu@192.222.50.72
```

**GPU**: NVIDIA GH200 96GB VRAM
**Currently Running**: 700M pretrain (step 51.5K/1M, 89% VRAM, ~2.4 days remaining)

### Key Directories
```
/home/ubuntu/nahawi/
├── gec_clean/epoch_11.pt           # Best content model (USE THIS)
├── fasih_v15_model/best_model.pt   # Most recent stable
├── fasih_v16_model/                # FAILED (3.56% F0.5)
├── ultimate_pretrain/              # 700M model training
├── corpus/combined/msa_corpus_full.txt  # 5.9GB elite MSA
├── data/qalb_real_train.json       # 37K real pairs
├── data/ultimate/                  # 66M synthetic
└── nahawi_spm.model                # 32K tokenizer
```

---

## LESSONS LEARNED

### Why V16 Failed
1. Synthetic data destroyed copy mechanism - hallucinated instead of copying
2. No identity pairs - never learned "correct text stays the same"
3. Wrong distribution - synthetic errors ≠ real Arabic errors

### Why Punct Is The Bottleneck
1. Model learns "add punct" but not "where"
2. Only 36% of generated punct in correct positions
3. 81% no-punct → 55% with-punct = 26 points lost to punct alone

### What We Now Know
1. **Content correction works** - 81% without punct is near-SOTA
2. **Punct needs separate handling** - not a seq2seq generation problem
3. **Data balance matters** - 1:1800 ratio teaches wrong patterns
4. **Real data anchors the model** - repeat QALB, don't drown it in synthetic

---

## NEXT STEPS

### Phase 1: Train New Base Model
The current approach (V15 + LoRA) reached 78.84%. To beat SOTA (82.63%), we need a stronger base.

**Options:**
1. **Train from scratch** with punct-aware data from the start
2. **Continue 700M pretrain** and fine-tune
3. **Architecture changes** (larger model, different attention)

### Phase 2: Finish FASIH Benchmark
Complete the exhaustive FASIH benchmark with all 18 categories.

---

## SUCCESS CRITERIA

| Metric | Current | Target | SOTA |
|--------|---------|--------|------|
| F0.5 with punct | **78.84%** | **83%+** | 82.63% |
| F0.5 no punct | 91.89% | maintain | ??? |
| FASIH accuracy | 85.93% | **90%+** | N/A |

**Gap to SOTA: 3.79 points** - achievable with better base model.

---

## QUICK COMMANDS

```bash
# SSH to server
ssh -i "C:\Users\Jesse\Downloads\nahawi.pem" ubuntu@192.222.50.72

# Check GPU
nvidia-smi

# Evaluate epoch 3 (best model)
python3 eval_epoch3_v2.py
```

---

## LOCAL FILES

### Key Directories
```
c:\nahawi\
├── models/punct_aware_lora/epoch_3.pt  # BEST checkpoint
├── benchmark/fasih/                     # FASIH benchmark
├── lora_experiment/                     # Training scripts
└── data/                                # Training data
```

### Training Scripts
- `lora_experiment/train_lora_punct_aware.py` - Punct-aware LoRA training
- `eval_epoch3_v2.py` - Evaluation script
- `eval_twopass.py` - Two-pass evaluation (proven ineffective)

---

*Last updated: January 11, 2025*
*Status: Epoch 3 is best (78.84%). Planning new base model training.*
