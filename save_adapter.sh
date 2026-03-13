#!/bin/bash
# =============================================================================
# Upload trained adapter to HuggingFace Hub
# Usage: bash /workspace/training_codellama/save_adapter.sh
# =============================================================================

HF_USERNAME="alisuleman525"
ADAPTER_DIR="/workspace/output/codellama-7b-python-adapter"
HF_REPO="${HF_USERNAME}/snake-repair-codellama-adapter"

echo "=============================================="
echo "  Uploading Adapter to HuggingFace Hub"
echo "=============================================="

if [ ! -f "${ADAPTER_DIR}/adapter_config.json" ]; then
    echo "  ERROR: No adapter at ${ADAPTER_DIR}"
    echo "  Run training first!"
    exit 1
fi

echo "  From: ${ADAPTER_DIR}"
echo "  To:   https://huggingface.co/${HF_REPO}"
echo ""
ls -lh "${ADAPTER_DIR}" | grep -v "checkpoint\|runs\|optimizer\|scheduler" | tail -n +2
echo ""

python3 -c "
from huggingface_hub import HfApi
api = HfApi()
api.create_repo('${HF_REPO}', private=True, exist_ok=True)
api.upload_folder(
    folder_path='${ADAPTER_DIR}',
    repo_id='${HF_REPO}',
    ignore_patterns=['checkpoint-*', 'runs/*', '*.log', 'optimizer.pt', 'scheduler.pt', 'rng_state*.pth'],
)
print('Done!')
"

echo ""
echo "  Uploaded to: https://huggingface.co/${HF_REPO}"
echo "  NOW STOP YOUR POD!"
