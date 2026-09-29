#!/bin/bash

#SBATCH -J move_results_to_home
#SBATCH -o /beegfs/scratch/callumdempsey/logs/job_move_to_home_%j.out
#SBATCH -e /beegfs/scratch/callumdempsey/logs/job_move_to_home_%j.err
#SBATCH -p ex_scioi_gpu
#SBATCH --mem=4G
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=2

set -e

SCRATCH_RESULTS="/beegfs/scratch/callumdempsey/results"
HOME_RESULTS="$HOME/results"
HOME_LOGS="$HOME/logs"

mkdir -p "$HOME_RESULTS"
mkdir -p "$HOME_LOGS"

echo "=== Step 1: building pruned checkpoints set (excluding tiny/Mock_Experiments) ==="
mkdir -p "$SCRATCH_RESULTS/pruned_checkpoints"
find "$SCRATCH_RESULTS/checkpoints" -maxdepth 1 -mindepth 1 \
    ! -iname "*tiny*" ! -iname "*Mock_Experiments*" \
    -exec cp -a {} "$SCRATCH_RESULTS/pruned_checkpoints/" \;
echo "Pruned set built. Size comparison:"
du -sh "$SCRATCH_RESULTS/checkpoints" "$SCRATCH_RESULTS/pruned_checkpoints"

echo "=== Step 2: rsyncing everything in results/ except embeddings and checkpoints ==="
rsync -avh --progress \
    --exclude="embeddings" \
    --exclude="checkpoints" \
    --exclude="pruned_checkpoints" \
    "$SCRATCH_RESULTS/" "$HOME_RESULTS/"

echo "=== Step 3: rsyncing pruned checkpoints into home as 'checkpoints' ==="
rsync -avh --progress \
    "$SCRATCH_RESULTS/pruned_checkpoints/" "$HOME_RESULTS/checkpoints/"

echo "=== Step 4: rsyncing logs ==="
rsync -avh --progress \
    "/beegfs/scratch/callumdempsey/logs/" "$HOME_LOGS/"

echo "=== Done. Verify before trusting it: ==="
echo "Source (scratch, excluding embeddings):"
du -sh --exclude=embeddings "$SCRATCH_RESULTS"
echo "Destination (home):"
du -sh "$HOME_RESULTS" "$HOME_LOGS"
