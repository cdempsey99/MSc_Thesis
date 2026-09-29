#!/bin/bash

#SBATCH -J profile_reben_step
#SBATCH -o /home/users/c/callumdempsey/logs/job_profile_%j.out
#SBATCH -e /home/users/c/callumdempsey/logs/job_profile_%j.err
#SBATCH -p ex_scioi_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=callum.dempsey@campus.tu-berlin.de

echo "=== Hardware Info ==="
echo "Node: $SLURMD_NODENAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Partition: $SLURM_JOB_PARTITION"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo "CPUs allocated: $SLURM_CPUS_ON_NODE"
echo "===================="

source /beegfs/scratch/callumdempsey/miniconda/bin/activate thesis_env
cd ~/MSc_Thesis

export REBEN=/beegfs/scratch/callumdempsey/data/reben

python3 -u profile_reben_training_step.py \
        --s2_root        "$REBEN/BigEarthNet-S2" \
        --ref_root       "$REBEN/Reference_Maps" \
        --metadata_path  "$REBEN/metadata.parquet" \
        --batch_size     16 \
        --n_unfrozen_blocks 4 \
        --decoder_embed_dim 512 \
        --num_classes    20 \
        --trace_out      /home/users/c/callumdempsey/results/profiler_trace_batch16.json
