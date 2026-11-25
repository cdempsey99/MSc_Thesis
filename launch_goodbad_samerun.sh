#!/bin/bash
# (1) Good/bad diversity split: re-evaluation with reeval_auroc.py (new fields disagreement_rate_on_errors /
#     _on_correct / disagreement_share_on_errors) of
#       FBP:   AS3 subset arms + JSD lambda sweep (one job), mains AS2/AV/AV+mask final (one job)
#       reBEN: AS3 subset arms (one job), mains AS2 / AV / AV+mask final (one job each)
#       SAR:   mains AS2 / AV / AV+mask final (one job)
#     Fresh tags (*_div), so no existing JSON is touched and --skip_done only matters for resubmissions.
# (2) Same-run LMC reference: linear_mode_connectivity.py between two checkpoints of ONE run
#     (epoch 40 vs final and epoch 20 vs final), for FBP and reBEN AS1 run1 and AS2 run1 (head 0).
# Checkpoints are resolved by pattern; any pattern matching != 1 file aborts before anything is submitted.
#   Dry run (default):  bash ~/MSc_Thesis/launch_goodbad_samerun.sh
#   Submit:             SUBMIT=1 bash ~/MSc_Thesis/launch_goodbad_samerun.sh

PY=/beegfs/scratch/callumdempsey/miniconda/envs/thesis_env/bin/python3
OUT=/home/users/c/callumdempsey/results
CK=${OUT}/checkpoints
REBEN=/beegfs/scratch/callumdempsey/data/reben
REBEN_SAR=/beegfs/scratch/callumdempsey/data/reben_SAR
FAIL=0

pick() {   # pick <regex>: the single file in $CK matching it
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
pick_all() {   # pick_all <regex> <expected count>: all matching files, count checked
    local m n
    m=$(ls "$CK" | grep -E -- "$1")
    n=$(printf '%s\n' "$m" | grep -c .)
    if [ "$n" -ne "$2" ]; then
        echo "ERROR: '$1' matched $n files, expected $2" >&2
        return 1
    fi
    echo "$m"
}
runs_of() {   # files -> "--run f1 --run f2 ..." (strip _decoder.pth for stems if asked)
    local strip=$1; shift
    for f in "$@"; do
        if [ "$strip" = "stem" ]; then echo -n " --run ${f%_decoder.pth}"; else echo -n " --run $f"; fi
    done
}

JOB_TAGS=(); JOB_TIMES=(); JOB_CMDS=()
queue() { JOB_TAGS+=("$1"); JOB_TIMES+=("$2"); JOB_CMDS+=("$3"); }

FBP_EVAL="reeval_auroc.py --pipeline fbp --data_dir /beegfs/scratch/callumdempsey/results --stride 224 --num_workers 16 --skip_done"
REBEN_EVAL="reeval_auroc.py --pipeline finetuned --clay_native_forward --clay_normalise_s2 --metadata_path ${REBEN}/metadata.parquet --s2_root ${REBEN}/BigEarthNet-S2 --ref_root ${REBEN}/Reference_Maps --num_workers 16 --skip_done"
SAR_EVAL="reeval_auroc.py --pipeline sar --clay_native_forward --metadata_path ${REBEN}/metadata.parquet --s1_root ${REBEN_SAR}/BigEarthNet-S1 --ref_root ${REBEN}/Reference_Maps --num_workers 16 --skip_done"

# ================= (1) good/bad diversity re-evaluation =================
# FBP AS3 arms (6 x 3) + lambda sweep (4 x 2) = 26 subset runs
F=$(pick_all '^FBP_AS3_subset75_M=5_40ep_.*_run[123]_.*_final_model_.*\.pth$' 26) || FAIL=1
queue FBP_AS3_div 12:00:00 "${FBP_EVAL} --max_images 112 --tag FBP_AS3_div$(runs_of file $F)"
# FBP mains: AS2 x3, AV x3, AV+mask x3
F=$(pick_all '^FBP_AS(2|4)_fulldataset_M=10_80ep_main_(run|av_run|av_mask_p0\.5_run)[123]_.*_final_model_.*\.pth$' 9) || FAIL=1
queue FBP_main_div 12:00:00 "${FBP_EVAL} --tag FBP_main_div$(runs_of file $F)"
# reBEN AS3 arms (6 x 3), subset test set
F=$(pick_all '^reBEN_AS3_subset25_M=5_40ep_.*_run[123]_.*_last_decoder\.pth$' 18) || FAIL=1
queue reBEN_AS3_div 12:00:00 "${REBEN_EVAL} --max_patches 60000 --tag reBEN_AS3_div$(runs_of stem $F)"
# reBEN mains, one job per stage
F=$(pick_all '^reBEN_AS2_fulldataset_M=10_80ep_main_run[123]_.*_last_decoder\.pth$' 3) || FAIL=1
queue reBEN_main_AS2_div 1-00:00:00 "${REBEN_EVAL} --tag reBEN_main_AS2_div$(runs_of stem $F)"
F=$(pick_all '^reBEN_AS4_fulldataset_M=10_80ep_main_av_run[123]_.*_last_decoder\.pth$' 3) || FAIL=1
queue reBEN_main_AV_div 1-00:00:00 "${REBEN_EVAL} --tag reBEN_main_AV_div$(runs_of stem $F)"
F=$(pick_all '^reBEN_AS4_fulldataset_M=10_80ep_main_av_mask_run[123]_.*_last_decoder\.pth$' 3) || FAIL=1
queue reBEN_main_AVM_div 1-00:00:00 "${REBEN_EVAL} --tag reBEN_main_AVM_div$(runs_of stem $F)"
# SAR mains: AS2 x3, AV x3, AV+mask x1
F=$(pick_all '^SAR_AS(2|4)_fulldataset_M=10_60ep_main_(run|av_run|av_plus_masking_run)[123]_.*_last_decoder\.pth$' 7) || FAIL=1
queue SAR_main_div 1-00:00:00 "${SAR_EVAL} --tag SAR_main_div$(runs_of stem $F)"

# ================= (2) same-run LMC reference =================
REBEN_DATA="--clay_native_forward --clay_normalise_s2 --metadata_path ${REBEN}/metadata.parquet --s2_root ${REBEN}/BigEarthNet-S2 --ref_root ${REBEN}/Reference_Maps"
for ST in "FBP_AS1_fulldataset_M=1_80ep_main_run1_" "FBP_AS2_fulldataset_M=10_80ep_main_run1_"; do
    FIN=$(pick "^${ST}.*_final_model_.*\.pth$") || FAIL=1
    short=$(echo "$ST" | sed -E 's/_fulldataset_M=[0-9]+_80ep_main_/_/; s/_$//')
    for E in 20 40; do
        SNAP=$(pick "^${ST}.*_ep${E}_model\.pth$") || FAIL=1
        queue "lmc_samerun_${short}_ep${E}_vs_final" 02:00:00 \
            "linear_mode_connectivity.py --pipeline fbp --run_a ${SNAP} --run_b ${FIN} --tag samerun_${short}_ep${E}_vs_final"
    done
done
for ST in "reBEN_AS1_fulldataset_M=1_80ep_main_run1_" "reBEN_AS2_fulldataset_M=10_80ep_main_run1_"; do
    FIN=$(pick "^${ST}.*_last_decoder\.pth$") || FAIL=1
    short=$(echo "$ST" | sed -E 's/_fulldataset_M=[0-9]+_80ep_main_/_/; s/_$//')
    for E in 20 40; do
        SNAP=$(pick "^${ST}.*_ep${E}_decoder\.pth$") || FAIL=1
        queue "lmc_samerun_${short}_ep${E}_vs_final" 06:00:00 \
            "linear_mode_connectivity.py --run_a ${SNAP%_decoder.pth} --run_b ${FIN%_decoder.pth} --tag samerun_${short}_ep${E}_vs_final ${REBEN_DATA}"
    done
done

# ================= check, then submit =================
if [ "$FAIL" -ne 0 ]; then
    echo "At least one checkpoint pattern did not match as expected (see ERROR lines above) - nothing submitted."
    exit 1
fi
for i in "${!JOB_TAGS[@]}"; do
    tag=${JOB_TAGS[$i]}
    if [ "${SUBMIT:-0}" = "1" ]; then
        sbatch -J "${tag}" \
            -p ex_scioi_gpu,gpu,scioi_gpu,gpu_short \
            --gres=gpu:v100s:1 \
            --mem=64G \
            --cpus-per-task=16 \
            --time="${JOB_TIMES[$i]}" \
            -o "$HOME/logs/${tag}_%j.out" \
            --wrap "cd $HOME/MSc_Thesis && export OUT_DIR=${OUT} && ${PY} -u ${JOB_CMDS[$i]}"
    else
        echo "[dry run] ${tag} (time ${JOB_TIMES[$i]}):"
        echo "${JOB_CMDS[$i]}" | sed -E 's/ --(run|tag|run_a|run_b) /
    -- /g'
    fi
done
echo "${#JOB_TAGS[@]} jobs."
