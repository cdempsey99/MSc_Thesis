#!/bin/bash
# Snapshot (learning-curve) re-evaluation of the final main runs (16/11/2026): reeval_auroc.py on the
# 20/40/60-epoch snapshots AND the final checkpoint of every main run, with calibration arrays.
#   FBP:   AS1 x3, AS2 x3, AS4 x4 (AV x3 + AV+mask hedge) - ep20, ep40, ep60, final   -> 3 jobs (one per stage)
#   reBEN: AS1 x3, AS2 x3, AS4 x4                          - ep20, ep40, ep60, _last   -> 10 jobs (one per run)
#   SAR:   AS1 x3, AS2 x1, AS4 x2 (60-ep runs)             - ep20, ep40, _last         -> 6 jobs (one per run)
# AS4 students are scored automatically from the same forward pass.
# The re-evaluated finals should reproduce the training-log numbers (sanity check for the snapshot numbers).
# Checkpoints are resolved by pattern; any pattern matching != 1 file aborts before anything is submitted.
#   Dry run (default, submits nothing):  bash ~/MSc_Thesis/launch_snapshot_reeval.sh
#   Submit:                              SUBMIT=1 bash ~/MSc_Thesis/launch_snapshot_reeval.sh
# A timed-out job can be resubmitted with the identical command (--skip_done skips finished checkpoints).
# Outputs: ~/results/runs/<tag>_auroc_reeval.json, ~/results/calibration/<dataset>_main_curve/*.npz,
#          log ~/logs/reeval_<tag>_<jobid>.out

PY=/beegfs/scratch/callumdempsey/miniconda/envs/thesis_env/bin/python3
OUT=/home/users/c/callumdempsey/results
CK=${OUT}/checkpoints
REBEN=/beegfs/scratch/callumdempsey/data/reben
REBEN_SAR=/beegfs/scratch/callumdempsey/data/reben_SAR
FAIL=0

# pick <regex>: the single file in $CK matching it
pick() {
    local m n
    m=$(ls "$CK" | grep -E -- "$1")
    n=$(printf '%s\n' "$m" | grep -c .)
    if [ "$n" -ne 1 ]; then
        echo "ERROR: '$1' matched $n files:" >&2
        printf '    %s\n' $m >&2
        return 1
    fi
    echo "$m"
}

# fbp_runs <run prefix>: appends --run for ep20/ep40/ep60 and the final model file to RUNS
fbp_runs() {
    local f s
    for s in ep20 ep40 ep60; do
        f=$(pick "^${1}.*_${s}_model\.pth$") || { FAIL=1; continue; }
        RUNS+=" --run ${f}"
    done
    f=$(pick "^${1}.*_final_model_.*\.pth$") || { FAIL=1; return; }
    RUNS+=" --run ${f}"
}

# e2e_runs <run prefix> <snapshot epochs...>: appends --run <stem> for each snapshot and _last to RUNS
e2e_runs() {
    local prefix=$1 f s
    shift
    for s in "$@" last; do
        f=$(pick "^${prefix}.*_${s}_decoder\.pth$") || { FAIL=1; continue; }
        RUNS+=" --run ${f%_decoder.pth}"
    done
}

JOB_TAGS=(); JOB_TIMES=(); JOB_ARGS=()
queue() {   # queue <tag> <time> <reeval_auroc.py args>
    JOB_TAGS+=("$1"); JOB_TIMES+=("$2"); JOB_ARGS+=("$3")
}

FBP_ARGS="--pipeline fbp \
--data_dir /beegfs/scratch/callumdempsey/results \
--stride 224 \
--num_workers 16 \
--skip_done \
--calibration_dir ${OUT}/calibration/FBP_main_curve"

REBEN_ARGS="--pipeline finetuned \
--clay_native_forward \
--clay_normalise_s2 \
--metadata_path ${REBEN}/metadata.parquet \
--s2_root ${REBEN}/BigEarthNet-S2 \
--ref_root ${REBEN}/Reference_Maps \
--num_workers 16 \
--skip_done \
--calibration_dir ${OUT}/calibration/reBEN_main_curve"

SAR_ARGS="--pipeline sar \
--clay_native_forward \
--metadata_path ${REBEN}/metadata.parquet \
--s1_root ${REBEN_SAR}/BigEarthNet-S1 \
--ref_root ${REBEN}/Reference_Maps \
--num_workers 16 \
--skip_done \
--calibration_dir ${OUT}/calibration/SAR_main_curve"

# ================= FBP: one job per stage =================
RUNS=""
for k in 1 2 3; do fbp_runs "FBP_AS1_fulldataset_M=1_80ep_main_run${k}_"; done
queue FBP_AS1_main_curve 12:00:00 "${FBP_ARGS} --tag FBP_AS1_main_curve${RUNS}"

RUNS=""
for k in 1 2 3; do fbp_runs "FBP_AS2_fulldataset_M=10_80ep_main_run${k}_"; done
queue FBP_AS2_main_curve 12:00:00 "${FBP_ARGS} --tag FBP_AS2_main_curve${RUNS}"

RUNS=""
for k in 1 2 3; do fbp_runs "FBP_AS4_fulldataset_M=10_80ep_main_av_run${k}_"; done
fbp_runs "FBP_AS4_fulldataset_M=10_80ep_main_av_mask_p0.5_run1_"
queue FBP_AS4_main_curve 12:00:00 "${FBP_ARGS} --tag FBP_AS4_main_curve${RUNS}"

# ================= reBEN: one job per run =================
for k in 1 2 3; do
    RUNS=""; e2e_runs "reBEN_AS1_fulldataset_M=1_80ep_main_run${k}_" ep20 ep40 ep60
    queue reBEN_AS1_main_run${k}_curve 1-00:00:00 "${REBEN_ARGS} --tag reBEN_AS1_main_run${k}_curve${RUNS}"
done
for k in 1 2 3; do
    RUNS=""; e2e_runs "reBEN_AS2_fulldataset_M=10_80ep_main_run${k}_" ep20 ep40 ep60
    queue reBEN_AS2_main_run${k}_curve 1-00:00:00 "${REBEN_ARGS} --tag reBEN_AS2_main_run${k}_curve${RUNS}"
done
for k in 1 2 3; do
    RUNS=""; e2e_runs "reBEN_AS4_fulldataset_M=10_80ep_main_av_run${k}_" ep20 ep40 ep60
    queue reBEN_AS4_main_av_run${k}_curve 1-00:00:00 "${REBEN_ARGS} --tag reBEN_AS4_main_av_run${k}_curve${RUNS}"
done
RUNS=""; e2e_runs "reBEN_AS4_fulldataset_M=10_80ep_main_av_mask_run1_" ep20 ep40 ep60
queue reBEN_AS4_main_avmask_run1_curve 1-00:00:00 "${REBEN_ARGS} --tag reBEN_AS4_main_avmask_run1_curve${RUNS}"

# ================= SAR: one job per run (60-epoch runs: ep20, ep40, _last) =================
for k in 1 2 3; do
    RUNS=""; e2e_runs "SAR_AS1_fulldataset_M=1_60ep_main_run${k}_" ep20 ep40
    queue SAR_AS1_main_run${k}_curve 1-00:00:00 "${SAR_ARGS} --tag SAR_AS1_main_run${k}_curve${RUNS}"
done
RUNS=""; e2e_runs "SAR_AS2_fulldataset_M=10_60ep_main_run1_" ep20 ep40
queue SAR_AS2_main_run1_curve 1-00:00:00 "${SAR_ARGS} --tag SAR_AS2_main_run1_curve${RUNS}"
RUNS=""; e2e_runs "SAR_AS4_fulldataset_M=10_60ep_main_av_run1_" ep20 ep40
queue SAR_AS4_main_av_run1_curve 1-00:00:00 "${SAR_ARGS} --tag SAR_AS4_main_av_run1_curve${RUNS}"
RUNS=""; e2e_runs "SAR_AS4_fulldataset_M=10_60ep_main_av_plus_masking_run1_" ep20 ep40
queue SAR_AS4_main_avmask_run1_curve 1-00:00:00 "${SAR_ARGS} --tag SAR_AS4_main_avmask_run1_curve${RUNS}"

# ================= check, then submit =================
if [ "$FAIL" -ne 0 ]; then
    echo "At least one checkpoint pattern did not match exactly one file (see ERROR lines above) - nothing submitted."
    exit 1
fi

for i in "${!JOB_TAGS[@]}"; do
    tag=${JOB_TAGS[$i]}
    if [ "${SUBMIT:-0}" = "1" ]; then
        sbatch -J "reeval_${tag}" \
            -p ex_scioi_gpu,gpu,scioi_gpu,gpu_short \
            --gres=gpu:v100s:1 \
            --mem=64G \
            --cpus-per-task=16 \
            --time="${JOB_TIMES[$i]}" \
            -o "$HOME/logs/reeval_${tag}_%j.out" \
            --wrap "cd $HOME/MSc_Thesis && export OUT_DIR=${OUT} && ${PY} -u reeval_auroc.py ${JOB_ARGS[$i]}"
    else
        echo "[dry run] reeval_${tag} (time ${JOB_TIMES[$i]}):"
        echo "${JOB_ARGS[$i]}" | sed -E 's/ --/\n    --/g'
    fi
done
echo "${#JOB_TAGS[@]} jobs."
