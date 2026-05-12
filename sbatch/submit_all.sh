#!/bin/bash
# Submit all six training regimens. Run from the repo root or anywhere.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for s in \
    train_nomask_nomt.sbatch \
    train_nomask_mt.sbatch \
    train_cellmask_nomt.sbatch \
    train_cellmask_mt.sbatch \
    train_nucmask_nomt.sbatch \
    train_nucmask_mt.sbatch
do
    echo "Submitting ${s}"
    sbatch "${SCRIPT_DIR}/${s}"
done
