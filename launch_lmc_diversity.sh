#!/bin/bash
# Linear mode connectivity on heads from the diversity-method runs (12/11/2026): 15 jobs.
#   AS3 subset runs, per dataset (FBP sub75 / reBEN subset, M=5, 40 ep):
#     control, JSD, fixed mask: head 0 vs head 1 within run1
#     control, AV, AV+mask:    head 2 of run1 vs run2 (head 2 of an M=5 AV run = 512 wide, 2 blocks = a default head)
#   Mains: FBP AS4 AV and reBEN AS4 AV, head 4 of run1 vs run2 (M=10 AV has no exact 512 head; 4 is mid-size),
#          plus the missing reBEN AS2 main run1 vs run2 head 0 baseline.
# Checkpoints are looked up by pattern at run time; any pattern matching != 1 file aborts before submitting.
#   Dry run (default, submits nothing, prints the resolved files):  bash ~/MSc_Thesis/launch_lmc_diversity.sh
#   Submit:                                                         SUBMIT=1 bash ~/MSc_Thesis/launch_lmc_diversity.sh
# Outputs: ~/results/runs/lmc_<tag>.json, ~/results/losses/lmc_<tag>.png, log ~/logs/lmc_<tag>_<jobid>.out

PY=/beegfs/scratch/callumdempsey/miniconda/envs/thesis_env/bin/python3
OUT=/home/users/c/callumdempsey/results
CK=${OUT}/checkpoints

# pick <include regex> <must-also-match regex or ''> <exclude regex or ''>  (case-insensitive, on file names in $CK)
pick() {
    local m
    m=$(ls "$CK" | grep -iE -- "$1")
    [ -n "$2" ] && m=$(printf '%s\n' "$m" | grep -iE -- "$2")
    [ -n "$3" ] && m=$(printf '%s\n' "$m" | grep -viE -- "$3")
    local n
    n=$(printf '%s\n' "$m" | grep -c .)
    if [ "$n" -ne 1 ]; then
        echo "ERROR: '$1' (+'$2', -'$3') matched $n files:" >&2
        printf '    %s\n' $m >&2
        echo MISSING
        return
    fi
    echo "$m"
}

# ---- FBP AS3 subset runs (final weights files) ----
FS='^FBP_AS[23]_subset75_M=5_40ep_'
FF='_final_model_.*\.pth$'
F_CTRL_1=$(pick  "${FS}.*_run1_.*${FF}" 'control'                 'param')
F_CTRL_2=$(pick  "${FS}.*_run2_.*${FF}" 'control'                 'param')
F_JSD_1=$(pick   "${FS}.*_run1_.*${FF}" 'jsd'                     'param')
F_MASK_1=$(pick  "${FS}.*_run1_.*${FF}" 'mask'                    'architecture|_av_|param')
F_AV_1=$(pick    "${FS}architecture_variation_run1_.*${FF}" ''    'mask')
F_AV_2=$(pick    "${FS}architecture_variation_run2_.*${FF}" ''    'mask')
F_AVM_1=$(pick   "${FS}.*_run1_.*${FF}" 'mask.*(architecture|_av)|(architecture|_av).*mask' 'param')
F_AVM_2=$(pick   "${FS}.*_run2_.*${FF}" 'mask.*(architecture|_av)|(architecture|_av).*mask' 'param')

# ---- reBEN AS3 subset runs (checkpoint stems = file name without _decoder.pth) ----
RS='^reBEN_AS[23]_'
RF='_last_decoder\.pth$'
R_CTRL_1=$(pick  "${RS}.*_run1_.*${RF}" 'control'                 'fulldataset|param')
R_CTRL_2=$(pick  "${RS}.*_run2_.*${RF}" 'control'                 'fulldataset|param')
R_JSD_1=$(pick   "${RS}.*_run1_.*${RF}" 'jsd'                     'fulldataset|param')
R_MASK_1=$(pick  "${RS}.*_run1_.*${RF}" 'mask'                    'fulldataset|architecture|_av_|param')
R_AV_1=$(pick    "${RS}.*_run1_.*${RF}" 'architecture'            'fulldataset|mask|param')
R_AV_2=$(pick    "${RS}.*_run2_.*${RF}" 'architecture'            'fulldataset|mask|param')
R_AVM_1=$(pick   "${RS}.*_run1_.*${RF}" 'mask.*(architecture|_av)|(architecture|_av).*mask' 'fulldataset|param')
R_AVM_2=$(pick   "${RS}.*_run2_.*${RF}" 'mask.*(architecture|_av)|(architecture|_av).*mask' 'fulldataset|param')

# ---- AS4 mains (AV, not the AV+mask hedge) and the reBEN AS2 main baseline ----
F_AS4_1=$(pick   "^FBP_AS4_.*_run1_.*${FF}"   ''                  'mask|qa|h200')
F_AS4_2=$(pick   "^FBP_AS4_.*_run2_.*${FF}"   ''                  'mask|qa|h200')
R_AS4_1=$(pick   "^reBEN_AS4_.*_run1_.*${RF}" ''                  'mask|qa|h200')
R_AS4_2=$(pick   "^reBEN_AS4_.*_run2_.*${RF}" ''                  'mask|qa|h200')
R_AS2_MAIN_1=reBEN_AS2_fulldataset_M=10_80ep_main_run1_20261106_1419_last_decoder.pth
R_AS2_MAIN_2=reBEN_AS2_fulldataset_M=10_80ep_main_run2_20261106_1420_last_decoder.pth

ALL="F_CTRL_1 F_CTRL_2 F_JSD_1 F_MASK_1 F_AV_1 F_AV_2 F_AVM_1 F_AVM_2 \
R_CTRL_1 R_CTRL_2 R_JSD_1 R_MASK_1 R_AV_1 R_AV_2 R_AVM_1 R_AVM_2 \
F_AS4_1 F_AS4_2 R_AS4_1 R_AS4_2 R_AS2_MAIN_1 R_AS2_MAIN_2"
echo "Resolved checkpoints:"
for v in $ALL; do printf '  %-13s %s\n' "$v" "${!v}"; done
for v in $ALL; do
    if [ "${!v}" = "MISSING" ]; then
        echo "At least one pattern did not match exactly one file (see ERROR lines above) - nothing submitted."
        exit 1
    fi
done
# reBEN: the script takes stems
for v in $ALL; do
    case $v in R_*) printf -v "$v" '%s' "${!v%_decoder.pth}" ;; esac
done

# submit <tag> <time limit> <linear_mode_connectivity.py args...>
submit() {
    local tag=$1 time=$2
    shift 2
    local cmd=(sbatch -J "lmc_${tag}"
        -p ex_scioi_gpu,gpu,scioi_gpu,gpu_short
        --gres=gpu:v100s:1
        --mem=32G
        --cpus-per-task=8
        --time="${time}"
        -o "$HOME/logs/lmc_${tag}_%j.out"
        --wrap "cd $HOME/MSc_Thesis && export OUT_DIR=${OUT} && ${PY} -u linear_mode_connectivity.py --tag ${tag} $*")
    if [ "${SUBMIT:-0}" = "1" ]; then
        "${cmd[@]}"
    else
        echo "[dry run] lmc_${tag}: linear_mode_connectivity.py --tag ${tag} $*"
    fi
}

REBEN_DATA="--clay_native_forward --clay_normalise_s2 \
--metadata_path /beegfs/scratch/callumdempsey/data/reben/metadata.parquet \
--s2_root /beegfs/scratch/callumdempsey/data/reben/BigEarthNet-S2 \
--ref_root /beegfs/scratch/callumdempsey/data/reben/Reference_Maps"

# ================= FBP AS3 subset (--max_images 112, as in training) =================
submit fbp_AS3_sub_control_run1_h0_vs_h1 02:00:00 \
    --pipeline fbp \
    --max_images 112 \
    --run_a ${F_CTRL_1} \
    --run_b ${F_CTRL_1} \
    --head_a 0 \
    --head_b 1

submit fbp_AS3_sub_jsd_run1_h0_vs_h1 02:00:00 \
    --pipeline fbp \
    --max_images 112 \
    --run_a ${F_JSD_1} \
    --run_b ${F_JSD_1} \
    --head_a 0 \
    --head_b 1

submit fbp_AS3_sub_mask_run1_h0_vs_h1 02:00:00 \
    --pipeline fbp \
    --max_images 112 \
    --run_a ${F_MASK_1} \
    --run_b ${F_MASK_1} \
    --head_a 0 \
    --head_b 1

submit fbp_AS3_sub_control_run1_vs_run2_h2 02:00:00 \
    --pipeline fbp \
    --max_images 112 \
    --run_a ${F_CTRL_1} \
    --run_b ${F_CTRL_2} \
    --head_a 2 \
    --head_b 2

submit fbp_AS3_sub_av_run1_vs_run2_h2 02:00:00 \
    --pipeline fbp \
    --max_images 112 \
    --run_a ${F_AV_1} \
    --run_b ${F_AV_2} \
    --head_a 2 \
    --head_b 2

submit fbp_AS3_sub_avmask_run1_vs_run2_h2 02:00:00 \
    --pipeline fbp \
    --max_images 112 \
    --run_a ${F_AVM_1} \
    --run_b ${F_AVM_2} \
    --head_a 2 \
    --head_b 2

# ================= reBEN AS3 subset =================
submit reben_AS3_sub_control_run1_h0_vs_h1 06:00:00 \
    --run_a ${R_CTRL_1} \
    --run_b ${R_CTRL_1} \
    --head_a 0 \
    --head_b 1 \
    ${REBEN_DATA}

submit reben_AS3_sub_jsd_run1_h0_vs_h1 06:00:00 \
    --run_a ${R_JSD_1} \
    --run_b ${R_JSD_1} \
    --head_a 0 \
    --head_b 1 \
    ${REBEN_DATA}

submit reben_AS3_sub_mask_run1_h0_vs_h1 06:00:00 \
    --run_a ${R_MASK_1} \
    --run_b ${R_MASK_1} \
    --head_a 0 \
    --head_b 1 \
    ${REBEN_DATA}

submit reben_AS3_sub_control_run1_vs_run2_h2 06:00:00 \
    --run_a ${R_CTRL_1} \
    --run_b ${R_CTRL_2} \
    --head_a 2 \
    --head_b 2 \
    ${REBEN_DATA}

submit reben_AS3_sub_av_run1_vs_run2_h2 06:00:00 \
    --run_a ${R_AV_1} \
    --run_b ${R_AV_2} \
    --head_a 2 \
    --head_b 2 \
    ${REBEN_DATA}

submit reben_AS3_sub_avmask_run1_vs_run2_h2 06:00:00 \
    --run_a ${R_AVM_1} \
    --run_b ${R_AVM_2} \
    --head_a 2 \
    --head_b 2 \
    ${REBEN_DATA}

# ================= Mains (full dataset) =================
submit fbp_AS4_full_av_run1_vs_run2_h4 02:00:00 \
    --pipeline fbp \
    --run_a ${F_AS4_1} \
    --run_b ${F_AS4_2} \
    --head_a 4 \
    --head_b 4

submit reben_AS4_full_av_run1_vs_run2_h4 06:00:00 \
    --run_a ${R_AS4_1} \
    --run_b ${R_AS4_2} \
    --head_a 4 \
    --head_b 4 \
    ${REBEN_DATA}

submit reben_AS2_full_run1_vs_run2_h0 06:00:00 \
    --run_a ${R_AS2_MAIN_1} \
    --run_b ${R_AS2_MAIN_2} \
    --head_a 0 \
    --head_b 0 \
    ${REBEN_DATA}
