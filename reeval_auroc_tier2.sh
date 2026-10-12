#!/bin/bash

#SBATCH -J reeval_auroc_tier2
#SBATCH -o /home/users/c/callumdempsey/logs/job_reeval_auroc_tier2_%j.out
#SBATCH -e /home/users/c/callumdempsey/logs/job_reeval_auroc_tier2_%j.err
#SBATCH -p ex_scioi_gpu,gpu,scioi_gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=callum.dempsey@campus.tu-berlin.de

# AUROC-only re-evaluation after the 08/10 AUROC sampling fix (see reeval_auroc.py).
# Usage: sbatch reeval_auroc_tier2.sh <group>
#   full_reben : full-scale reBEN optical AS1 / AS2 / AS4 (AS4 = AS3 ensemble + student)
#   full_sar   : full-scale reBEN SAR AS1 / AS2 (SAR AS3/AS4 get fixed AUROCs from their running resumes)
# Acc/mIoU are printed per run as a check that the right checkpoint was loaded - they must match the
# historical numbers (reBEN AS1 .6680/.3935, AS2 .6724/.4014; SAR AS1 .6054/.3345, AS2 .6116/.3436).

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
export REBEN_SAR=/beegfs/scratch/callumdempsey/data/reben_SAR

case "$1" in
    full_reben)
        python3 -u reeval_auroc.py --pipeline finetuned --tag full_reben \
            --s2_root "$REBEN/BigEarthNet-S2" --ref_root "$REBEN/Reference_Maps" --metadata_path "$REBEN/metadata.parquet" \
            --run AS1_reben_qa_full_resume_20260706_1108_best \
            --run AS2_finetuned_m=10_full_dataset_run_20260715_1801_best \
            --run AS4_finetuned_m=10_diverse_fulldataset_student_resumed4_20260817_1123_last
        ;;
    full_sar)
        python3 -u reeval_auroc.py --pipeline sar --tag full_sar \
            --s1_root "$REBEN_SAR/BigEarthNet-S1" --ref_root "$REBEN/Reference_Maps" --metadata_path "$REBEN/metadata.parquet" \
            --run AS1_reben_sar_qa_full_dataset_resumed_20260715_1227_best \
            --run AS2_reben_sar_m10_fulldataset_resumed4_20260924_1632_last
        ;;
    *)
        echo "Unknown group '$1' (expected full_reben or full_sar)"; exit 1
        ;;
esac
