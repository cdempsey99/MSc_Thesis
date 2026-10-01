#!/bin/bash

#SBATCH -J profile_fbp
#SBATCH -o /home/users/c/callumdempsey/logs/job_profile_fbp_%j.out
#SBATCH -e /home/users/c/callumdempsey/logs/job_profile_fbp_%j.err
#SBATCH -p ex_scioi_gpu,scioi_gpu
#SBATCH --gres=gpu:v100s:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=callum.dempsey@campus.tu-berlin.de

echo "=== Hardware Info ==="
echo "Node: $SLURMD_NODENAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Partition: $SLURM_JOB_PARTITION"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo "===================="

source /beegfs/scratch/callumdempsey/miniconda/bin/activate thesis_env
cd ~/MSc_Thesis

export EMBEDDINGS_DIR="/beegfs/scratch/callumdempsey/results"
export TRACES="/home/users/c/callumdempsey/results"

# V100S so the numbers compare with the 2.19 s/batch fp32 baseline (job 1998252).
# Three variants in one job so it only queues once; each runs independently.
echo "########## fp32 (current training setup) ##########"
python3 -u profile_fbp_training_step.py --data_dir $EMBEDDINGS_DIR \
        --trace_out $TRACES/profiler_trace_fbp_fp32.json

echo "########## --compile ##########"
python3 -u profile_fbp_training_step.py --data_dir $EMBEDDINGS_DIR --compile \
        --trace_out $TRACES/profiler_trace_fbp_compile.json

echo "########## --amp ##########"
python3 -u profile_fbp_training_step.py --data_dir $EMBEDDINGS_DIR --amp \
        --trace_out $TRACES/profiler_trace_fbp_amp.json
