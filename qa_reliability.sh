#!/bin/bash

#SBATCH -J qa_reliability
#SBATCH -o /home/users/c/callumdempsey/logs/job_qa_reliability_%j.out
#SBATCH -e /home/users/c/callumdempsey/logs/job_qa_reliability_%j.err
#SBATCH -p ex_scioi_gpu,gpu,scioi_gpu
#SBATCH --gres=gpu:v100s:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=callum.dempsey@campus.tu-berlin.de

# Tiny QA for the 22/10 calibration changes (results are throwaway, QA_ run names):
#   1. reBEN fine-tuned, 160 train / 32 test patches, 1 epoch -> new training-time reliability plot (train_e2e_reben.py)
#   2. FBP, 3 images (1 train / 1 val / 1 test) / 1 epoch / M=2          -> new reliability plot from utils.misc.evaluate_test_set
#   3. reeval_auroc.py --calibration_dir on one existing reBEN checkpoint (32 test patches) + plot_calibration.py
# Check afterwards: the job log has no tracebacks; ~/results/reliability/QA_reliability_*.png are two-panel
# (accuracy at bin mean confidence + pixel/error share bars); ~/results/calibration_plots/qa/ has the 4-panel
# figure and calibration_summary.csv; the reeval log shows "rho ... Q ..." in the diversity line.

echo "=== Hardware Info ==="
echo "Node: $SLURMD_NODENAME"
echo "Job ID: $SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo "===================="

source /beegfs/scratch/callumdempsey/miniconda/bin/activate thesis_env
cd ~/MSc_Thesis

export OUT_DIR="/home/users/c/callumdempsey/results"
export SCRATCH_DATA="/beegfs/scratch/callumdempsey/data/fbp"
export REBEN=/beegfs/scratch/callumdempsey/data/reben
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "### 1. reBEN fine-tuned training-time reliability plot"
python3 -u train_e2e_reben.py \
        --s2_root "$REBEN/BigEarthNet-S2" --ref_root "$REBEN/Reference_Maps" --metadata_path "$REBEN/metadata.parquet" \
        --run_name QA_reliability_reben --num_classes 20 --ensemble_size 5 --n_unfrozen_blocks 4 \
        --num_epochs 1 --warmup_epochs 1 --batch_size 32 --num_workers 16 --lr_encoder 1e-6 --lr_decoder 1e-4 \
        --lr_schedule cosine --use_focal_loss --hide_unlabelled_pixels --no_mean_logit_loss --max_patches 160 \
        --head_mask_prob 0.9 --head_mask_mode fixed --head_mask_seed 0

echo "### 2. FBP evaluate_test_set reliability plot"
python3 -u train_decoders.py \
        --data_dir /beegfs/scratch/callumdempsey/results --out_dir "$OUT_DIR" --run_name QA_reliability_fbp \
        --decoder_in_channels 1024 --decoder_embed_dim 512 --ensemble_size 2 --num_classes 25 --num_epochs 1 \
        --batch_size 32 --num_workers 4 --hide_unlabelled_pixels --stride 224 --use_focal_loss \
        --eval_checkpoint final --max_images 3

echo "### 3. reeval_auroc.py --calibration_dir + plot_calibration.py"
python3 -u reeval_auroc.py --pipeline finetuned --tag qa_calibration --max_patches 160 \
        --calibration_dir "$OUT_DIR/calibration/qa" \
        --s2_root "$REBEN/BigEarthNet-S2" --ref_root "$REBEN/Reference_Maps" --metadata_path "$REBEN/metadata.parquet" \
        --run AS2_reben_baseline_LR_cosine_run1_20261005_1541_best
python3 -u plot_calibration.py "$OUT_DIR"/calibration/qa/qa_calibration__*__teacher.npz \
        --labels "AS2 run1 (QA, 32 patches)" --classes reben --out_dir "$OUT_DIR/calibration_plots/qa"

echo "### QA done"
