#!/bin/bash
# Post-hoc temperature scaling of the AS1 main runs (16/11/2026): 9 jobs.
#   FBP AS1 x3   -> calibrate_as1_frozen.py (T fitted on val, image-level bootstrap of T)
#   reBEN AS1 x3 -> calibrate_as1_e2e.py --dataset reben
#   SAR AS1 x3   -> calibrate_as1_e2e.py --dataset sar
# T is fitted on the validation split only and applied to test; accuracy/mIoU/AUROC are unchanged by
# construction, only ECE/NLL move. reBEN/SAR also save raw + calibrated arrays for plot_calibration.py.
#   Dry run (default, submits nothing):  bash ~/MSc_Thesis/launch_temperature_scaling.sh
#   Submit:                              SUBMIT=1 bash ~/MSc_Thesis/launch_temperature_scaling.sh
# Logs: ~/logs/ts_<run_name>_<jobid>.out

PY=/beegfs/scratch/callumdempsey/miniconda/envs/thesis_env/bin/python3
OUT=/home/users/c/callumdempsey/results
CK=${OUT}/checkpoints
REBEN=/beegfs/scratch/callumdempsey/data/reben
REBEN_SAR=/beegfs/scratch/callumdempsey/data/reben_SAR

# submit <run_name> <time> <script> <args>
submit() {
    local name=$1 time=$2 script=$3
    shift 3
    if [ "${SUBMIT:-0}" = "1" ]; then
        sbatch -J "ts_${name}" \
            -p ex_scioi_gpu,gpu,scioi_gpu,gpu_short \
            --gres=gpu:v100s:1 \
            --mem=64G \
            --cpus-per-task=16 \
            --time="${time}" \
            -o "$HOME/logs/ts_${name}_%j.out" \
            --wrap "cd $HOME/MSc_Thesis && export OUT_DIR=${OUT} && ${PY} -u ${script} --run_name ${name} $*"
    else
        echo "[dry run] ts_${name} (time ${time}): ${script}"
        echo "    --run_name ${name} $*" | sed -E 's/ --/\n    --/g; s/^\n//'
    fi
}

# ================= FBP (frozen; script defaults are stride 112 / embed 256, so both set explicitly) =================
FBP_ARGS="--data_dir /beegfs/scratch/callumdempsey/results \
--patch_size 224 \
--stride 224 \
--decoder_embed_dim 512 \
--num_classes 25 \
--bootstrap_n 100"

submit FBP_AS1_main_run1_TS 03:00:00 calibrate_as1_frozen.py \
    --decoder_checkpoint ${CK}/FBP_AS1_fulldataset_M=1_80ep_main_run1_20261103_1901_final_model_20261104_0810.pth \
    ${FBP_ARGS}

submit FBP_AS1_main_run2_TS 03:00:00 calibrate_as1_frozen.py \
    --decoder_checkpoint ${CK}/FBP_AS1_fulldataset_M=1_80ep_main_run2_20261103_1907_final_model_20261104_0820.pth \
    ${FBP_ARGS}

submit FBP_AS1_main_run3_TS 03:00:00 calibrate_as1_frozen.py \
    --decoder_checkpoint ${CK}/FBP_AS1_fulldataset_M=1_80ep_main_run3_20261103_2109_final_model_20261104_1130.pth \
    ${FBP_ARGS}

# ================= reBEN (fine-tuned encoder + decoder, final = _last) =================
REBEN_ARGS="--dataset reben \
--metadata_path ${REBEN}/metadata.parquet \
--img_root ${REBEN}/BigEarthNet-S2 \
--ref_root ${REBEN}/Reference_Maps \
--n_unfrozen_blocks 4 \
--decoder_embed_dim 512 \
--max_val_patches 61000 \
--clay_native_forward \
--clay_normalise_s2 \
--calibration_dir ${OUT}/calibration/reBEN_AS1_main_TS"

for k in 1 2 3; do
    case $k in 1) S=reBEN_AS1_fulldataset_M=1_80ep_main_run1_20261105_0042_last ;;
               2) S=reBEN_AS1_fulldataset_M=1_80ep_main_run2_20261105_0059_last ;;
               3) S=reBEN_AS1_fulldataset_M=1_80ep_main_run3_20261105_0121_last ;; esac
    submit reBEN_AS1_main_run${k}_TS 12:00:00 calibrate_as1_e2e.py \
        --decoder_checkpoint ${CK}/${S}_decoder.pth \
        --encoder_checkpoint ${CK}/${S}_encoder.pth \
        ${REBEN_ARGS}
done

# ================= SAR (no S2 normalisation flag) =================
SAR_ARGS="--dataset sar \
--metadata_path ${REBEN}/metadata.parquet \
--img_root ${REBEN_SAR}/BigEarthNet-S1 \
--ref_root ${REBEN}/Reference_Maps \
--n_unfrozen_blocks 4 \
--decoder_embed_dim 512 \
--max_val_patches 61000 \
--clay_native_forward \
--calibration_dir ${OUT}/calibration/SAR_AS1_main_TS"

for k in 1 2 3; do
    case $k in 1) S=SAR_AS1_fulldataset_M=1_60ep_main_run1_20261105_0228_last ;;
               2) S=SAR_AS1_fulldataset_M=1_60ep_main_run2_20261105_0252_last ;;
               3) S=SAR_AS1_fulldataset_M=1_60ep_main_run3_20261105_0915_last ;; esac
    submit SAR_AS1_main_run${k}_TS 12:00:00 calibrate_as1_e2e.py \
        --decoder_checkpoint ${CK}/${S}_decoder.pth \
        --encoder_checkpoint ${CK}/${S}_encoder.pth \
        ${SAR_ARGS}
done
