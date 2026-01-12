#!/bin/bash
# LoRA Experiment - Run all steps

set -e

echo "========================================"
echo "LORA EXPERIMENT"
echo "========================================"
echo "Testing hypothesis: data distribution vs capacity"
echo ""

cd /home/ubuntu/nahawi

# Step 1: Create stratified 500K dataset
echo "Step 1: Creating stratified 500K dataset..."
python3 lora_experiment/sample_stratified_500k.py

# Step 2: Train LoRA model
echo ""
echo "Step 2: Training LoRA model (rank 32)..."
python3 lora_experiment/train_lora.py

# Step 3: Evaluate LoRA model (content only)
echo ""
echo "Step 3: Evaluating LoRA model..."
python3 lora_experiment/eval_lora.py

# Step 4: Train punct classifier
echo ""
echo "Step 4: Training punct classifier..."
python3 lora_experiment/train_punct_classifier.py

# Step 5: Final evaluation with punct
echo ""
echo "Step 5: Final evaluation..."
python3 lora_experiment/eval_lora.py

echo ""
echo "========================================"
echo "EXPERIMENT COMPLETE"
echo "========================================"
