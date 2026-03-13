#!/bin/bash
# =============================================================================
# Snake-RepairLLaMA: Training Launcher
# =============================================================================
# Usage:
#   tmux new -s training
#   bash /workspace/training_codellama/train.sh
# =============================================================================

set -e

export HF_HOME=/workspace/huggingface_cache

REPO_DIR="/workspace/training_codellama"
DATASET_DIR="/workspace/dataset"
OUTPUT_DIR="/workspace/output/codellama-7b-python-adapter"
LOG_FILE="/workspace/training.log"

# --- Check tmux ---
if [ -z "$TMUX" ]; then
    echo "WARNING: Not inside tmux! Training will stop if SSH disconnects."
    echo "  Recommended: tmux new -s training && bash $0"
    read -p "Continue anyway? (y/N): " confirm
    [ "$confirm" != "y" ] && [ "$confirm" != "Y" ] && exit 0
fi

# --- Auto-detect GPU and set batch size ---
GPU_MEM=$(python3 -c "import torch; print(int(torch.cuda.get_device_properties(0).total_mem / 1e9))" 2>/dev/null || echo "0")
GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "Unknown")

echo "=============================================="
echo "  Snake-RepairLLaMA Training"
echo "=============================================="
echo "  GPU: ${GPU_NAME} (${GPU_MEM}GB)"

if [ "$GPU_MEM" -ge 40 ]; then
    BATCH=8; GRAD_ACCUM=2
    echo "  Mode: A6000/A100 (batch=8, grad_accum=2)"
elif [ "$GPU_MEM" -ge 20 ]; then
    BATCH=4; GRAD_ACCUM=4
    echo "  Mode: RTX 4090 (batch=4, grad_accum=4)"
else
    BATCH=1; GRAD_ACCUM=16
    echo "  Mode: Low VRAM (batch=1, grad_accum=16)"
fi
echo "  Effective batch: $((BATCH * GRAD_ACCUM))"
echo "  Output: ${OUTPUT_DIR}"
echo "  Log: ${LOG_FILE}"
echo "=============================================="
echo ""

python3 "${REPO_DIR}/train_adapter.py" \
    --model_name_or_path codellama/CodeLlama-7b-Python-hf \
    --train_data_path "${DATASET_DIR}/train.parquet" \
    --validation_data_path "${DATASET_DIR}/validation.parquet" \
    --output_dir "${OUTPUT_DIR}" \
    --per_device_train_batch_size ${BATCH} \
    --per_device_eval_batch_size ${BATCH} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --num_train_epochs 3 \
    --learning_rate 5e-4 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --model_max_length 1024 \
    --fp16 True \
    --logging_steps 50 \
    --evaluation_strategy steps \
    --eval_steps 500 \
    --save_strategy steps \
    --save_steps 500 \
    --save_total_limit 5 \
    --report_to tensorboard \
    --dataloader_pin_memory True \
    --dataloader_num_workers 4 \
    2>&1 | tee "${LOG_FILE}"

echo ""
echo "=============================================="
echo "  Training Complete!"
echo "=============================================="
echo "  Adapter: ${OUTPUT_DIR}"
echo "  Upload:  bash ${REPO_DIR}/save_adapter.sh"
echo "  STOP THE POD to save money!"
echo ""
