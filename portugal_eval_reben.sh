#!/bin/bash

#SBATCH -J portugal_eval_reben
#SBATCH -o /home/users/c/callumdempsey/logs/job_portugal_eval_%j.out
#SBATCH -e /home/users/c/callumdempsey/logs/job_portugal_eval_%j.err
#SBATCH -p ex_scioi_gpu,gpu,scioi_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=callum.dempsey@campus.tu-berlin.de

# Held-out-country (Portugal) test, fine-tuned reBEN arch var, cosine, 25k subset. Checkpoint-only, no training.
# 2x2 grid, every model evaluated on its own:
#   rows:    trained WITHOUT Portugal (noPT run1/run2)  vs  trained WITH Portugal (cosine arch var run1/run3/run4/run2b)
#   columns: Portuguese test patches (--test_countries)  vs  non-Portuguese test patches (--exclude_test_countries)
# --exclude_test_countries Portugal + --max_patches 25000 rebuilds exactly the noPT runs' own test set, so the
# noPT x nonPT cells are also the AUROC-fixed re-eval of noPT run1 (trained before the AUROC fix).
# Cosine arch var run2 (20261005_1744) is excluded: it timed out on gpu066 at epoch 16.

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

NOPT="AS3_reben_architecture_var_noPT_run1_20261008_1505 AS3_reben_architecture_var_noPT_run2_20261008_1815"
WITHPT="AS2_reben_architecture_var_LR_cosine_run1_20261005_1617 AS2_reben_architecture_var_LR_cosine_run3_20261005_1819 AS2_reben_architecture_var_LR_cosine_run4_20261005_1847 AS3_reben_architecture_var_LR_cosine_run2b_20261007_0032"

# Each eval is skipped if its results JSON already exists, so a resubmission after a timeout only
# runs the missing ones (first attempt, job 2019214, timed out after 5 of 12 on 12/10).
RES=$OUT_DIR/runs
for RUN in $NOPT $WITHPT; do
    for SPLIT in onPT nonPT; do
        NAME=PT_EVAL_${RUN%%_2026*}_${SPLIT}
        if [ -f "$RES/${NAME}_results.json" ]; then echo "skip $NAME (done)"; continue; fi
        if [ $SPLIT = onPT ]; then FILT="--test_countries Portugal"; else FILT="--exclude_test_countries Portugal"; fi
        python3 -u posthoc_ensemble_reben.py $COMMON --member $RUN $FILT --run_name $NAME
    done
done
