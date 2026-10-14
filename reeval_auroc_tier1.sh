#!/bin/bash

#SBATCH -J reeval_auroc_tier1
#SBATCH -o /home/users/c/callumdempsey/logs/job_reeval_auroc_tier1_%j.out
#SBATCH -e /home/users/c/callumdempsey/logs/job_reeval_auroc_tier1_%j.err
#SBATCH -p ex_scioi_gpu,gpu,scioi_gpu
#SBATCH --gres=gpu:v100s:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=callum.dempsey@campus.tu-berlin.de

# AUROC-only re-evaluation (see reeval_auroc.py) of the two reBEN reduced-scale rounds (25k/5k/5k):
# the fine-tuned cosine batch (05-07/10) and the frozen-encoder batch (07/10).
# Usage: sbatch reeval_auroc_tier1.sh <group>
#   subset_ft     : 24 fine-tuned cosine runs
#   subset_frozen : 15 frozen-encoder runs
# Excluded: AS2 baseline run3 (20261005_1555) and run4 (20261005_1606), arch var run2 (20261005_1744) - all timed
# out on gpu066 mid-training; the tiny_qa runs. NB the cosine arch var run1/3/4 carry an AS2_ prefix by mistake.

echo "=== Hardware Info ==="
echo "Node: $SLURMD_NODENAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Partition: $SLURM_JOB_PARTITION"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo "CPUs allocated: $SLURM_CPUS_ON_NODE"
echo "Group: $1"
echo "===================="

source /beegfs/scratch/callumdempsey/miniconda/bin/activate thesis_env
cd ~/MSc_Thesis

export OUT_DIR="/home/users/c/callumdempsey/results"
export REBEN=/beegfs/scratch/callumdempsey/data/reben
EMBED_DIR=/beegfs/scratch/callumdempsey/results/embeddings/reben

case "$1" in
    subset_ft)
        RUNS=""
        for r in \
            AS1_reben_LR_cosine_run1_20261007_0610 \
            AS1_reben_LR_cosine_run2_20261007_0715 \
            AS1_reben_LR_cosine_run3_20261007_0716 \
            AS2_reben_baseline_LR_cosine_run1_20261005_1541 \
            AS2_reben_baseline_LR_cosine_run2_20261005_1542 \
            AS2_reben_baseline_LR_cosine_run3b_20261006_2317 \
            AS2_reben_baseline_LR_cosine_run4b_20261006_2317 \
            AS2_reben_architecture_var_LR_cosine_run1_20261005_1617 \
            AS2_reben_architecture_var_LR_cosine_run3_20261005_1819 \
            AS2_reben_architecture_var_LR_cosine_run4_20261005_1847 \
            AS3_reben_architecture_var_LR_cosine_run2b_20261007_0032 \
            AS3_reben_bootstrap_mask_batch_LR_cosine_run1_20261005_1848 \
            AS3_reben_bootstrap_mask_batch_LR_cosine_run2_20261005_1918 \
            AS3_reben_bootstrap_mask_batch_LR_cosine_run3_20261005_2107 \
            AS3_reben_bootstrap_mask_batch_LR_cosine_run4_20261005_2124 \
            AS3_reben_bootstrap_mask_fixed_LR_cosine_run1_20261005_2134 \
            AS3_reben_bootstrap_mask_fixed_LR_cosine_run2_20261005_2219 \
            AS3_reben_bootstrap_mask_fixed_LR_cosine_run3_20261005_2225 \
            AS3_reben_bootstrap_mask_fixed_LR_cosine_run4_20261005_2344 \
            AS3_reben_bootstrap_mask_fixed_LR_cosine_seed1_run5_20261007_0040 \
            AS3_reben_bootstrap_mask_fixed_LR_cosine_seed2_run6_20261007_0156 \
            AS3_reben_bootstrap_mask_fixed_plus_arch_var_LR_cosine_run1_20261007_0321 \
            AS3_reben_bootstrap_mask_fixed_plus_arch_var_LR_cosine_run2_20261007_0410 \
            AS3_reben_bootstrap_mask_fixed_plus_arch_var_LR_cosine_run3_20261007_0431 ; do
            RUNS="$RUNS --run ${r}_best"
        done
        python3 -u reeval_auroc.py --pipeline finetuned --tag subset_ft --max_patches 25000 \
            --s2_root "$REBEN/BigEarthNet-S2" --ref_root "$REBEN/Reference_Maps" --metadata_path "$REBEN/metadata.parquet" \
            $RUNS
        ;;
    subset_frozen)
        RUNS=""
        for r in \
            AS1_frozen_reben_run1_20261007_1220 \
            AS1_frozen_reben_run2_20261007_1220 \
            AS1_frozen_reben_run3_20261007_1220 \
            AS2_frozen_reben_run1_20261007_1221 \
            AS2_frozen_reben_run2_20261007_1221 \
            AS2_frozen_reben_run3_20261007_1221 \
            AS3_frozen_reben_architecture_var_run1_20261007_1222 \
            AS3_frozen_reben_architecture_var_run2_20261007_1222 \
            AS3_frozen_reben_architecture_var_run3_20261007_1222 \
            AS3_frozen_reben_bootstrap_masking_fixed_run1_20261007_1224 \
            AS3_frozen_reben_bootstrap_masking_fixed_run2_20261007_1224 \
            AS3_frozen_reben_bootstrap_masking_fixed_run3_20261007_1224 \
            AS3_frozen_reben_jsd_run1_20261007_1225 \
            AS3_frozen_reben_jsd_run2_20261007_1225 \
            AS3_frozen_reben_jsd_run3_20261007_1346 ; do
            RUNS="$RUNS --run ${r}_best"
        done
        python3 -u reeval_auroc.py --pipeline frozen --tag subset_frozen --max_patches 25000 \
            --embedding_dir "$EMBED_DIR" $RUNS
        ;;
    *)
        echo "Unknown group '$1' (expected subset_ft or subset_frozen)"; exit 1
        ;;
esac
