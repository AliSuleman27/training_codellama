#!/bin/bash
# =============================================================================
# Snake-RepairLLaMA: Resume Training from Checkpoint
# =============================================================================
# Use after pod restart/preemption.
# Usage:
#   tmux new -s training
#   bash /workspace/training_codellama/resume.sh
# =============================================================================

set -e

export HF_HOME=/workspace/huggingface_cache

REPO_DIR="/workspace/training_codellama"
DATASET_DIR="/workspace/dataset"
OUTPUT_DIR="/workspace/output/codellama-7b-python-adapter"
LOG_FILE="/workspace/training_resume.log"

echo "=============================================="
echo "  Snake-RepairLLaMA: Resume Training"
echo "=============================================="

# --- Re-install deps if needed (container disk resets on restart) ---
echo ""
echo "[1/3] Checking dependencies..."
python3 -c "import peft, bitsandbytes, transformers" 2>/dev/null || {
    echo "  Re-installing (container disk was reset)..."
    pip install -q -r "${REPO_DIR}/requirements.txt"
}
echo "  Dependencies OK."

# --- Find latest checkpoint ---
echo ""
echo "[2/3] Finding latest checkpoint..."
LATEST_CKPT=$(ls -d ${OUTPUT_DIR}/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)

if [ -z "$LATEST_CKPT" ]; then
    echo "  No checkpoints found! Starting fresh..."
    exec bash "${REPO_DIR}/train.sh"
fi

CKPT_STEP=$(basename "$LATEST_CKPT" | grep -o '[0-9]*')
echo "  Resuming from: checkpoint-${CKPT_STEP}"
echo ""
echo "  All checkpoints:"
ls -d ${OUTPUT_DIR}/checkpoint-* 2>/dev/null | while read ckpt; do
    echo "    $(basename $ckpt) ($(du -sh "$ckpt" 2>/dev/null | cut -f1))"
done

# --- Auto-detect GPU ---
GPU_MEM=$(python3 -c "import torch; print(int(torch.cuda.get_device_properties(0).total_mem / 1e9))" 2>/dev/null || echo "0")
GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "Unknown")

if [ "$GPU_MEM" -ge 40 ]; then BATCH=8; GRAD_ACCUM=2;
elif [ "$GPU_MEM" -ge 20 ]; then BATCH=4; GRAD_ACCUM=4;
else BATCH=1; GRAD_ACCUM=16; fi

echo ""
echo "[3/3] Resuming..."
echo "  GPU: ${GPU_NAME} (${GPU_MEM}GB), batch=${BATCH}, grad_accum=${GRAD_ACCUM}"
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
    --resume_from_checkpoint "${LATEST_CKPT}" \
    2>&1 | tee "${LOG_FILE}"

echo ""
echo "  Training resumed and completed!"
