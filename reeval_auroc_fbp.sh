#!/bin/bash

#SBATCH -J reeval_auroc_fbp
#SBATCH -o /home/users/c/callumdempsey/logs/job_reeval_auroc_fbp_%j.out
#SBATCH -e /home/users/c/callumdempsey/logs/job_reeval_auroc_fbp_%j.err
#SBATCH -p ex_scioi_gpu,gpu,scioi_gpu
#SBATCH --gres=gpu:v100s:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=callum.dempsey@campus.tu-berlin.de

# FBP re-evaluation (see reeval_auroc.py --pipeline fbp): fixed AUROC sample + epistemic/aleatoric AUROC +
# diversity metrics (normalised disagreement, error correlation ratio, oracle acc, unanimous errors) for the
# existing FBP checkpoints. Full dataset: 150 images, stride 224 -> the same 23-image test split as training.
# Usage: sbatch reeval_auroc_fbp.sh <group>
#   fbp_main     : AS1-AS4 main table, 20 and 80 epochs (end-of-training weights)
#   fbp_variants : 08/09 diversity variants (hyperparam var, arch var, per-head variational layer, bagging)
#   fbp_ablation : ensemble size M = 3/5/7/10/12 x 20/40/60 epochs (AS4 runs)
#   fbp_lambda   : JSD lambda sweep (runs that trained cleanly to the end)
# Resubmitting the same group skips checkpoints already in its JSON (--skip_done).

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
COMMON="--pipeline fbp --data_dir /beegfs/scratch/callumdempsey/results --stride 224 --max_images 150 --num_workers 4 --skip_done"

case "$1" in
    fbp_main)
        # The 20 / 80 epoch main table: end-of-training weights of the full-dataset runs. AS1 has two final_model
        # files (16/06 = end of the original run, 16/07 = a later resume/re-eval); both scored, mIoU shows if they
        # differ. AS3 20ep = the corrected 21/09 rerun, i.e. the checkpoint the 80ep _p2 run was extended from.
        RUNS=""
        for r in \
            AS1_class_imbalance_focal_loss_qa_150img_20260616_1443_final_model_20260616_1953.pth \
            AS1_class_imbalance_focal_loss_qa_150img_20260616_1443_final_model_20260716_1127.pth \
            AS2_FBP_frozen_encoder_m=10_ensemble_20260715_1755_final_model_20260716_0208.pth \
            AS2_FBP_frozen_encoder_m=10_ensemble_extended80_final_model_20260926_1512.pth \
            AS3_frozen_encoder_m=10_ensemble_corrected_20260921_1543_final_model_20260923_1317.pth \
            AS3_fbp_fulldataset_m=10_ensemble_corrected_extended80_p2_final_model_20261014_0640.pth \
            AS4_fbp_frozen_encoder_fulldataset_m10_student_training_20260727_1626_final_model_20260728_1958.pth \
            AS4_fbp_fulldataset_m=10_ensemble_corrected_extended80_p2_final_model_20261014_0415.pth ; do
            RUNS="$RUNS --run $r"
        done
        python3 -u reeval_auroc.py $COMMON --tag fbp_main $RUNS
        ;;
    fbp_variants)
        # 08/09 diversity variants (M=5, 40ep), end-of-training weights (final_model), as in fbp_main.
        # Arch var and VB lamkl0.01/sigma0.5 each have two final_model files (later resume/re-eval): both scored.
        RUNS=""
        for r in \
            AS3_fbp_hyperparam_var_40ep_20260904_1443_final_model_20260906_0848.pth \
            AS3_fbp_architecture_var_40ep_20260904_1556_final_model_20260906_1133.pth \
            AS3_fbp_architecture_var_40ep_20260904_1556_final_model_20260908_0820.pth \
            AS3_fbp_bagging_40_ep_20260910_1719_final_model_20260911_1852.pth \
            AS3_fbp_architecture_var_bagging_40_ep_20260910_1648_final_model_20260911_1808.pth \
            AS3_fbp_variational_bottleneck_v2_lamkl0.001_sigmaprior0.5_20260904_1327_final_model_20260906_0752.pth \
            AS3_fbp_variational_bottleneck_v2_lamkl0.01_sigmaprior0.5_20260831_1152_20260831_1156_final_model_20260901_1446.pth \
            AS3_fbp_variational_bottleneck_v2_lamkl0.01_sigmaprior0.5_20260831_1152_20260831_1156_final_model_20260905_1416.pth \
            AS3_fbp_variational_bottleneck_v2_lamkl0.01_sigmaprior1.0_20260902_1118_final_model_20260904_0636.pth \
            AS3_fbp_variational_bottleneck_v2_lamkl0.05_sigmaprior1.0_20260828_1859_final_model_20260829_1752.pth \
            AS3_fbp_variational_bottleneck_v2_lamkl0.1_sigmaprior0.5_20260902_1118_final_model_20260904_0636.pth \
            AS3_fbp_variational_bottleneck_v2_lamkl0.1_sigmaprior1.0_20260831_1203_20260831_1208_final_model_20260901_1451.pth \
            AS3_fbp_variational_bottleneck_v2_lamkl1.0_sigmaprior1.0_20260828_1920_final_model_20260829_1740.pth \
            AS4_fbp_architecture_var_variational_bottleneck_20260907_1427_final_model_20260911_2252.pth ; do
            RUNS="$RUNS --run $r"
        done
        python3 -u reeval_auroc.py $COMMON --tag fbp_variants $RUNS
        ;;
    fbp_ablation)
        # Ensemble size x training length (AS4: teacher + in-parallel student, JSD lam 0.1), end-of-training weights.
        # The 40/60ep files are resumed extensions from before the student-reinitialisation fix: teachers valid,
        # students NOT (the student was re-initialised at the resume). M=10 20ep is also in fbp_main.
        RUNS=""
        for r in \
            AS4_fbp_frozen_encoder_fulldataset_m3_student_training_20260803_1344_final_model_20260803_2006.pth \
            AS4_fbp_frozen_encoder_fulldataset_m5_student_training_20260731_1648_final_model_20260801_0054.pth \
            AS4_fbp_frozen_encoder_fulldataset_m5_student_training_extended_final_model_20260803_2246.pth \
            AS4_fbp_frozen_encoder_fulldataset_m7_student_training_20260729_1748_final_model_20260731_0233.pth \
            AS4_fbp_frozen_encoder_fulldataset_m7_student_training_extended_final_model_20260804_0406.pth \
            AS4_fbp_frozen_encoder_fulldataset_m10_student_training_20260727_1626_final_model_20260728_1958.pth \
            AS4_fbp_frozen_encoder_fulldataset_m10_student_training_extended_final_model_20260801_0251.pth \
            AS4_fbp_frozen_encoder_fulldataset_m10_student_training_extended_extended_final_model_20260805_0352.pth \
            AS4_fbp_frozen_encoder_fulldataset_m12_student_training_20260731_1649_final_model_20260802_0839.pth \
            AS4_fbp_frozen_encoder_fulldataset_m12_student_training_extended_final_model_20260807_0945.pth ; do
            RUNS="$RUNS --run $r"
        done
        python3 -u reeval_auroc.py $COMMON --tag fbp_ablation $RUNS
        ;;
    fbp_lambda)
        # JSD lambda sweep, runs that trained cleanly to the end: lam 0.3 (unclipped, 06/08; plus its 13/08
        # eval-final re-save) and lam 0.05 (clipped, 07/08), both M=5 40ep (lam 0.1 at M=5 40ep = the M=5
        # extended run in fbp_ablation); and M=10 lam 0.3 at 20 and 40ep. The runs with lam > 0.3 all went NaN
        # mid-training, so only early pre-crash snapshots exist for them - left out.
        RUNS=""
        for r in \
            AS3_fbp_varying_lambda_0.3_20260806_1808_final_model_20260807_0654.pth \
            AS3_fbp_varying_lambda_0.3_evalfinal_final_model_20260813_1613.pth \
            AS3_fbp_varying_lambda_clipped_0.05_20260807_1311_final_model_20260808_0627.pth \
            AS3_fbp_varying_m=10_ensemble_lam=0.3_20260820_1529_final_model_20260822_1257.pth \
            AS3_fbp_varying_m=10_ensemble_lam=0.3_40_ep_final_model_20260826_0856.pth ; do
            RUNS="$RUNS --run $r"
        done
        python3 -u reeval_auroc.py $COMMON --tag fbp_lambda $RUNS
        ;;
    *)
        echo "Unknown group '$1' (expected fbp_main, fbp_variants, fbp_ablation or fbp_lambda)"; exit 1
        ;;
esac
