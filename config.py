"""
Snake-RepairLLaMA: Centralized Configuration for RunPod Training
Edit this file ONCE before running setup.sh
"""

# =============================================================================
# HuggingFace Configuration
# =============================================================================
HF_USERNAME = "alisuleman525"
HF_DATASET_REPO = f"{HF_USERNAME}/snake-repair-data"
HF_ADAPTER_REPO = f"{HF_USERNAME}/snake-repair-codellama-adapter"

# =============================================================================
# Model Configuration
# =============================================================================
BASE_MODEL = "codellama/CodeLlama-7b-Python-hf"

# =============================================================================
# Paths (on RunPod - all under /workspace for persistence)
# =============================================================================
WORKSPACE = "/workspace"
HF_CACHE_DIR = f"{WORKSPACE}/huggingface_cache"
DATASET_DIR = f"{WORKSPACE}/dataset"
TRAIN_DATA = f"{DATASET_DIR}/train.parquet"
VAL_DATA = f"{DATASET_DIR}/validation.parquet"
OUTPUT_DIR = f"{WORKSPACE}/output/codellama-7b-python-adapter"
REPO_DIR = f"{WORKSPACE}/SnakeRepair-LLAMA"

# =============================================================================
# Training Hyperparameters
# =============================================================================
# --- GPU-specific presets ---
# A6000 (48GB) / A100 (40GB): batch=8, grad_accum=2 -> effective batch 16
# RTX 4090 (24GB):             batch=4, grad_accum=4 -> effective batch 16
# T4 (16GB) / Colab:           batch=1, grad_accum=16 -> effective batch 16

BATCH_SIZE = 8                  # Change to 4 for RTX 4090
GRAD_ACCUMULATION = 2           # Change to 4 for RTX 4090
LEARNING_RATE = 5e-4
NUM_EPOCHS = 3
MAX_SEQ_LENGTH = 512            # Covers 95%+ of samples (mean ~281 tokens)
WARMUP_RATIO = 0.03
LR_SCHEDULER = "cosine"

# =============================================================================
# LoRA Configuration
# =============================================================================
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj"]

# =============================================================================
# Checkpointing (critical for RunPod - don't lose progress)
# =============================================================================
SAVE_STEPS = 500
EVAL_STEPS = 500
SAVE_TOTAL_LIMIT = 5
LOGGING_STEPS = 50

# =============================================================================
# Quantization
# =============================================================================
USE_8BIT = True                 # INT8 for training (good balance)
# Note: Colab used 4-bit because T4 only has 16GB
# On A6000 (48GB), 8-bit gives better quality and still fits easily
