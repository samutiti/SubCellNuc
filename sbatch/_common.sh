# Sourced by every train_*.sbatch. Edit these once to retarget the cluster.

# Path to the data directory (single-cell crops).
export DATA_DIR="/scratch/users/samutiti/U54/data/subcell_XAP_images"

# Where checkpoints are written. Each sbatch appends a regimen suffix.
export OUT_ROOT="/scratch/users/samutiti/U54/runs/cnn"

# Python env. Match the existing data_gen/make_subcell_esm_data.sbatch pattern.
load_env() {
    module load python/3.12
    # Adjust this path if your virtualenv lives elsewhere.
    source "${HOME}/env/bin/activate"
}

# Common training hyperparameters. Override per-sbatch by setting EXTRA_ARGS.
export EPOCHS=50
export BATCH_SIZE=128
export NUM_WORKERS=8
export IMAGE_SIZE=128
export LR=3e-4
