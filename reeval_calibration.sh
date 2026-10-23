#!/bin/bash

#SBATCH -J reeval_calibration
#SBATCH -o /home/users/c/callumdempsey/logs/job_reeval_calibration_%j.out
#SBATCH -e /home/users/c/callumdempsey/logs/job_reeval_calibration_%j.err
#SBATCH -p ex_scioi_gpu,gpu,scioi_gpu
#SBATCH --gres=gpu:v100s:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=callum.dempsey@campus.tu-berlin.de

# Calibration detail for the error-confidence curves Ronny asked for (22/10): reeval_auroc.py --calibration_dir
# saves per-checkpoint arrays, then plot_calibration.py makes the figures + calibration_summary.csv.
# Usage: sbatch reeval_calibration.sh <group>
#   reben_ft : fine-tuned reBEN 25k subset, run 1 of AS1 / AS2 / fixed mask p0.5 / p0.9 / JSD (the checkpoints
#              the reported numbers came from)
#   fbp      : FBP AS1 20ep + AS1 / AS2 / AS3 / AS4 (teacher + student) at 80ep, final weights

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
CAL_DIR="$OUT_DIR/calibration/$1"

case "$1" in
    reben_ft)
        python3 -u reeval_auroc.py --pipeline finetuned --tag calib_reben_ft --max_patches 25000 --skip_done \
            --calibration_dir "$CAL_DIR" \
            --s2_root "$REBEN/BigEarthNet-S2" --ref_root "$REBEN/Reference_Maps" --metadata_path "$REBEN/metadata.parquet" \
            --run AS1_reben_LR_cosine_run1_20261007_0610_best \
            --run AS2_reben_baseline_LR_cosine_run1_20261005_1541_best \
            --run AS3_reben_bootstrap_mask_fixed_LR_cosine_run1_20261005_2134_best \
            --run AS3_reben_cosine_fixedmask_p09_run1_20261021_1902_best \
            --run AS3_reben_cosine_jsd_run1_20261014_1540_best
        python3 -u plot_calibration.py "$CAL_DIR"/calib_reben_ft__AS1_*__teacher.npz \
            "$CAL_DIR"/calib_reben_ft__AS2_*__teacher.npz \
            "$CAL_DIR"/calib_reben_ft__AS3_reben_bootstrap_mask_fixed_*__teacher.npz \
            "$CAL_DIR"/calib_reben_ft__AS3_reben_cosine_fixedmask_p09_*__teacher.npz \
            "$CAL_DIR"/calib_reben_ft__AS3_reben_cosine_jsd_*__teacher.npz \
            --labels "AS1" "AS2" "Fixed mask p=0.5" "Fixed mask p=0.9" "JSD" \
            --classes reben --out_dir "$OUT_DIR/calibration_plots/reben_ft"
        ;;
    fbp)
        python3 -u reeval_auroc.py --pipeline fbp --tag calib_fbp --skip_done --calibration_dir "$CAL_DIR" \
            --embedding_dir /beegfs/scratch/callumdempsey/results/embeddings/fbp/clay_v1/patch224_stride224_old_bkp \
            --stride 224 --max_images 150 --num_workers 4 \
            --run AS1_class_imbalance_focal_loss_qa_150img_20260616_1443_final_model_20260616_1953.pth \
            --run AS1_fbp_fulldataset_extended80_final_model_20261021_0736.pth \
            --run AS2_FBP_frozen_encoder_m=10_ensemble_extended80_final_model_20260926_1512.pth \
            --run AS3_fbp_fulldataset_m=10_ensemble_corrected_extended80_p2_final_model_20261014_0640.pth \
            --run AS4_fbp_fulldataset_m=10_ensemble_corrected_extended80_p2_final_model_20261014_0415.pth
        python3 -u plot_calibration.py \
            "$CAL_DIR"/calib_fbp__AS1_class_imbalance_*__teacher.npz \
            "$CAL_DIR"/calib_fbp__AS1_fbp_fulldataset_extended80_*__teacher.npz \
            "$CAL_DIR"/calib_fbp__AS2_*__teacher.npz \
            "$CAL_DIR"/calib_fbp__AS3_*__teacher.npz \
            "$CAL_DIR"/calib_fbp__AS4_*__teacher.npz \
            "$CAL_DIR"/calib_fbp__AS4_*__student.npz \
            --labels "AS1 20ep" "AS1 80ep" "AS2 80ep" "AS3 80ep" "AS4 teacher 80ep" "AS4 student 80ep" \
            --classes fbp --out_dir "$OUT_DIR/calibration_plots/fbp"
        ;;
    *)
        echo "Unknown group '$1' (expected reben_ft or fbp)"; exit 1
        ;;
esac
