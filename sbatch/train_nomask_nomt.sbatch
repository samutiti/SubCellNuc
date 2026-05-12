#!/bin/bash
#SBATCH --job-name=cnn_nomask_nomt
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail
mkdir -p logs

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"
load_env

REGIMEN="nomask_nomt"
OUT_DIR="${OUT_ROOT}/${REGIMEN}"

cd "${SCRIPT_DIR}/.."
python -m cnn.train \
    --data_dir "${DATA_DIR}" \
    --mask_mode none \
    --use_mt 0 \
    --image_size "${IMAGE_SIZE}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --epochs "${EPOCHS}" \
    --lr "${LR}" \
    --out_dir "${OUT_DIR}" \
    --amp 1
