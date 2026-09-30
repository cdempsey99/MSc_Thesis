#!/bin/bash

#SBATCH -J combine_reben_sar
#SBATCH -o /home/users/c/callumdempsey/logs/job_combine_reben_sar_%j.out
#SBATCH -e /home/users/c/callumdempsey/logs/job_combine_reben_sar_%j.err
#SBATCH -p ex_scioi_gpu
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=callum.dempsey@campus.tu-berlin.de

echo "=== Hardware Info ==="
echo "Node: $SLURMD_NODENAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Partition: $SLURM_JOB_PARTITION"
echo "CPUs allocated: $SLURM_CPUS_ON_NODE"
echo "===================="

source /beegfs/scratch/callumdempsey/miniconda/bin/activate thesis_env
cd ~/MSc_Thesis

export REBEN=/beegfs/scratch/callumdempsey/data/reben
export REBEN_SAR=/beegfs/scratch/callumdempsey/data/reben_SAR

# CPU-only (no --gres): reads 2 band files and writes 1 combined file per patch. Full dataset,
# all splits - check the "skipped due to missing band files" count on each "Done with" line.
python3 -u combine_input_files_reben_sar.py \
        --s1_root        "$REBEN_SAR/BigEarthNet-S1" \
        --metadata_path  "$REBEN/metadata.parquet" \
        --num_workers    16
