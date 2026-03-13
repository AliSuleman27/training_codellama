#!/bin/bash
# =============================================================================
# Snake-RepairLLaMA: RunPod Full Setup Script
# =============================================================================
# Run this ONCE when you first start your RunPod pod.
#
# Usage:
#   git clone https://github.com/AliSuleman27/training_codellama.git /workspace/training_codellama
#   bash /workspace/training_codellama/setup.sh
# =============================================================================

set -e

# --- Configuration ---
HF_USERNAME="alisuleman525"
HF_DATASET_REPO="${HF_USERNAME}/snake-repair-data"
REPO_DIR="/workspace/training_codellama"
DATASET_DIR="/workspace/dataset"
OUTPUT_DIR="/workspace/output/codellama-7b-python-adapter"
HF_CACHE="/workspace/huggingface_cache"

echo ""
echo "=============================================="
echo "  Snake-RepairLLaMA: RunPod Setup"
echo "=============================================="
echo ""

# -------------------------------------------------
# Step 1: Persistent HuggingFace cache
# -------------------------------------------------
echo "[1/5] Setting up persistent HF cache..."
export HF_HOME="${HF_CACHE}"
mkdir -p "${HF_CACHE}" "${DATASET_DIR}" "${OUTPUT_DIR}"
grep -q "HF_HOME" ~/.bashrc 2>/dev/null || echo "export HF_HOME=${HF_CACHE}" >> ~/.bashrc
echo "  HF_HOME=${HF_CACHE}"

# -------------------------------------------------
# Step 2: HuggingFace Login
# -------------------------------------------------
echo ""
echo "[2/5] HuggingFace Login..."
echo "  Get a token at: https://huggingface.co/settings/tokens"
python3 -c "from huggingface_hub import login; login()" 2>/dev/null || {
    echo "  Deferring login to after deps install..."
    NEED_LOGIN=1
}

# -------------------------------------------------
# Step 3: Install Python dependencies
# -------------------------------------------------
echo ""
echo "[3/5] Installing Python dependencies..."
pip install --upgrade pip -q
pip install -q -r "${REPO_DIR}/requirements.txt"

if [ "${NEED_LOGIN}" = "1" ]; then
    echo ""
    echo "  Logging into HuggingFace now..."
    python3 -c "from huggingface_hub import login; login()"
fi

# Verify
echo "  Verifying..."
python3 -c "import bitsandbytes; print('    bitsandbytes: OK')" || {
    pip install bitsandbytes --force-reinstall --no-cache-dir -q
    python3 -c "import bitsandbytes; print('    bitsandbytes: OK (reinstalled)')"
}
python3 -c "import peft; print('    peft: OK')"
python3 -c "import transformers; print('    transformers: OK')"
python3 -c "import torch; print(f'    torch: OK (CUDA: {torch.cuda.is_available()})')"

# -------------------------------------------------
# Step 4: Download dataset from HuggingFace Hub
# -------------------------------------------------
echo ""
echo "[4/5] Downloading dataset..."
if [ -f "${DATASET_DIR}/train.parquet" ] && [ -f "${DATASET_DIR}/validation.parquet" ]; then
    echo "  Dataset already exists:"
    echo "    train.parquet: $(du -h ${DATASET_DIR}/train.parquet | cut -f1)"
    echo "    validation.parquet: $(du -h ${DATASET_DIR}/validation.parquet | cut -f1)"
else
    python3 -c "
from huggingface_hub import hf_hub_download
print('  Downloading train.parquet...')
hf_hub_download('${HF_DATASET_REPO}', 'train.parquet', repo_type='dataset', local_dir='${DATASET_DIR}/')
print('  Downloading validation.parquet...')
hf_hub_download('${HF_DATASET_REPO}', 'validation.parquet', repo_type='dataset', local_dir='${DATASET_DIR}/')
print('  Done!')
"
fi

# -------------------------------------------------
# Step 5: Download base model
# -------------------------------------------------
echo ""
echo "[5/5] Downloading CodeLlama-7B-Python (~13GB)..."
python3 -c "
from transformers import AutoTokenizer
try:
    t = AutoTokenizer.from_pretrained('codellama/CodeLlama-7b-Python-hf', local_files_only=True)
    print('  Model already cached!')
except:
    print('  Downloading model...')
    from huggingface_hub import snapshot_download
    snapshot_download('codellama/CodeLlama-7b-Python-hf')
    t = AutoTokenizer.from_pretrained('codellama/CodeLlama-7b-Python-hf')
print(f'  Tokenizer vocab: {len(t)}')
"

# -------------------------------------------------
# Final verification
# -------------------------------------------------
echo ""
echo "=============================================="
echo "  Verification"
echo "=============================================="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo "  No GPU!"
df -h /workspace | tail -1 | awk '{print "  Disk: " $2 " total, " $4 " free"}'
python3 -c "
from datasets import load_dataset
train = load_dataset('parquet', data_files='${DATASET_DIR}/train.parquet', split='train')
val = load_dataset('parquet', data_files='${DATASET_DIR}/validation.parquet', split='train')
print(f'  Dataset: {len(train)} train, {len(val)} val samples')
"

echo ""
echo "=============================================="
echo "  Setup Complete!"
echo "=============================================="
echo ""
echo "  Start training:"
echo "    tmux new -s training"
echo "    bash ${REPO_DIR}/train.sh"
echo ""
echo "  Resume from checkpoint:"
echo "    bash ${REPO_DIR}/resume.sh"
echo ""
