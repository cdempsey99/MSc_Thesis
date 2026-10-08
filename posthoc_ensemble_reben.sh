#!/bin/bash

#SBATCH -J posthoc_ensemble_reben
#SBATCH -o /home/users/c/callumdempsey/logs/job_posthoc_%j.out
#SBATCH -e /home/users/c/callumdempsey/logs/job_posthoc_%j.err
#SBATCH -p ex_scioi_gpu,gpu,scioi_gpu,gpu_short
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=callum.dempsey@campus.tu-berlin.de

# Post-hoc ensemble evaluation (fine-tuned reBEN, 25k subset). Checkpoint-only, no training.
# 1. sanity check, 2. 3 separately fine-tuned AS1 runs as one ensemble, 3. M-matched control:
# 3-of-5 head subsets of jointly trained AS2 runs. Do NOT use AS2 run3/run4 (20261005_1555/_1606):
# those timed out on gpu066, so their best checkpoints are from incomplete training.

echo "=== Hardware Info ==="
echo "Node: $SLURMD_NODENAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Partition: $SLURM_JOB_PARTITION"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo "CPUs allocated: $SLURM_CPUS_ON_NODE"
echo "===================="

source /beegfs/scratch/callumdempsey/miniconda/bin/activate thesis_env
cd ~/MSc_Thesis

export OUT_DIR="/home/users/c/callumdempsey/results"
export REBEN=/beegfs/scratch/callumdempsey/data/reben
COMMON="--pipeline finetuned --s2_root $REBEN/BigEarthNet-S2 --ref_root $REBEN/Reference_Maps --metadata_path $REBEN/metadata.parquet --max_patches 25000 --n_unfrozen_blocks 4"

# 1. Sanity: a single run on its own must reproduce its logged numbers
#    (AS1 run1 log: mIoU 0.2702 | ECE 0.0738 | NLL 0.7319 | AUROC 0.6191)
python3 -u posthoc_ensemble_reben.py $COMMON \
    --member AS1_reben_LR_cosine_run1_20261007_0610 \
    --run_name POSTHOC_sanity_ft_AS1run1

# 2. Main test: 3 separately fine-tuned AS1 runs as one ensemble
python3 -u posthoc_ensemble_reben.py $COMMON \
    --member AS1_reben_LR_cosine_run1_20261007_0610 \
    --member AS1_reben_LR_cosine_run2_20261007_0715 \
    --member AS1_reben_LR_cosine_run3_20261007_0716 \
    --run_name POSTHOC_ft_AS1x3

# 3. M-matched control: 3 of the 5 heads of jointly trained AS2 runs
for RUN in AS2_reben_baseline_LR_cosine_run1_20261005_1541 AS2_reben_baseline_LR_cosine_run4b_20261006_2317; do
    for H in 0,1,2 2,3,4 0,2,4; do
        python3 -u posthoc_ensemble_reben.py $COMMON \
            --member ${RUN}:${H} \
            --run_name POSTHOC_ft_${RUN%%_2026*}_heads${H//,/}
    done
done
