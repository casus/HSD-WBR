#!/usr/bin/env bash
set -euo pipefail

# --- 1. Environment & Hardware Settings for Apple Silicon (M1 Max MPS) ---
export PYTORCH_ENABLE_MPS_FALLBACK=1

RESULTS_DIR="examples/mobilesam/models"
mkdir -p "${RESULTS_DIR}/logs"
mkdir -p "weights"

# Fetch lightweight MobileSAM weights if not present (~40 MB)
WEIGHTS_PATH="weights/mobile_sam.pt"
if [ ! -f "$WEIGHTS_PATH" ]; then
    echo "Downloading pretrained MobileSAM (vit_t) weights..."
    curl -L "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt" -o "$WEIGHTS_PATH"
fi

# --- 2. Training (Baseline vs. WBR) across Replicate Seeds ---
SEEDS=(11 42 123)

for SEED in "${SEEDS[@]}"; do
    echo "=========================================================="
    echo "Starting MobileSAM Baseline (lambda_3=0.0) | Seed: ${SEED}"
    echo "=========================================================="
    python scripts/mobilesam/train_mobilesam.py \
        configs/sample_train_mobilesam_baseline_config.json \
        --seed "${SEED}" \
        > "${RESULTS_DIR}/logs/mobilesam_baseline_seed_${SEED}.log" 2>&1

    echo "=========================================================="
    echo "Starting MobileSAM + WBR (lambda_3=1.0)   | Seed: ${SEED}"
    echo "=========================================================="
    python scripts/mobilesam/train_mobilesam.py \
        configs/sample_train_mobilesam_wbr_config.json \
        --seed "${SEED}" \
        > "${RESULTS_DIR}/logs/mobilesam_wbr_seed_${SEED}.log" 2>&1
done

# --- 3. Save Predictions (Test Set Inference) ---
echo "=========================================================="
echo "Generating test set mask predictions..."
echo "=========================================================="
python scripts/mobilesam/save_predictions_mobilesam.py \
    configs/sample_save_pred_mobilesam_config.json \
    > "${RESULTS_DIR}/logs/save_predictions.log" 2>&1

# --- 4. Evaluate JTPR & Containment Violation ---
echo "=========================================================="
echo "Evaluating containment violation and IoU metrics..."
echo "=========================================================="
python scripts/mobilesam/evaluate_mobilesam.py \
    configs/sample_jtpr_eval_mobilesam_config.json \
    > "${RESULTS_DIR}/logs/containment_eval.log" 2>&1

echo "=========================================================="
echo "Evaluating Joint True Positive Rate (JTPR)..."
echo "=========================================================="
python scripts/cellpose/evaluate_new_metrics_cellpose.py \
    configs/sample_jtpr_eval_mobilesam_config.json \
    > "${RESULTS_DIR}/logs/jtpr_eval.log" 2>&1

echo "=========================================================="
echo "MobileSAM experiment pipeline completed successfully."
echo "Check logs in: ${RESULTS_DIR}/logs/"
echo "=========================================================="

