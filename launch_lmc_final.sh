#!/bin/bash
# Linear mode connectivity on the final main runs (11/11/2026): 9 jobs, one per pair.
#   reBEN AS1 run-vs-run (3), reBEN AS2 heads within run1 (1),
#   FBP AS1 run-vs-run (3), FBP AS2 heads within run1 + head 0 run1-vs-run2 (2).
# Run from the frontend after git pull:  bash ~/MSc_Thesis/launch_lmc_final.sh
# Outputs: ~/results/runs/lmc_<tag>.json, ~/results/losses/lmc_<tag>.png, log ~/logs/lmc_<tag>_<jobid>.out

PY=/beegfs/scratch/callumdempsey/miniconda/envs/thesis_env/bin/python3
OUT=/home/users/c/callumdempsey/results

# submit <tag> <time limit> <linear_mode_connectivity.py args...>
submit() {
    local tag=$1 time=$2
    shift 2
    sbatch -J "lmc_${tag}" \
        -p ex_scioi_gpu,gpu,scioi_gpu,gpu_short \
        --gres=gpu:v100s:1 \
        --mem=32G \
        --cpus-per-task=8 \
        --time="${time}" \
        -o "$HOME/logs/lmc_${tag}_%j.out" \
        --wrap "cd $HOME/MSc_Thesis && export OUT_DIR=${OUT} && ${PY} -u linear_mode_connectivity.py --tag ${tag} $*"
}

REBEN_DATA="--clay_native_forward --clay_normalise_s2 \
--metadata_path /beegfs/scratch/callumdempsey/data/reben/metadata.parquet \
--s2_root /beegfs/scratch/callumdempsey/data/reben/BigEarthNet-S2 \
--ref_root /beegfs/scratch/callumdempsey/data/reben/Reference_Maps"

R1=reBEN_AS1_fulldataset_M=1_80ep_main_run1_20261105_0042_last
R2=reBEN_AS1_fulldataset_M=1_80ep_main_run2_20261105_0059_last
R3=reBEN_AS1_fulldataset_M=1_80ep_main_run3_20261105_0121_last
R_AS2=reBEN_AS2_fulldataset_M=10_80ep_main_run1_20261106_1419_last

F1=FBP_AS1_fulldataset_M=1_80ep_main_run1_20261103_1901_final_model_20261104_0810.pth
F2=FBP_AS1_fulldataset_M=1_80ep_main_run2_20261103_1907_final_model_20261104_0820.pth
F3=FBP_AS1_fulldataset_M=1_80ep_main_run3_20261103_2109_final_model_20261104_1130.pth
F_AS2_1=FBP_AS2_fulldataset_M=10_80ep_main_run1_20261103_2207_final_model_20261104_1625.pth
F_AS2_2=FBP_AS2_fulldataset_M=10_80ep_main_run2_20261103_2256_final_model_20261104_1725.pth

# ---- reBEN (fine-tuned encoder + decoder interpolated; ~3.2 h on a P100, less on V100S) ----
submit reben_AS1_full_run1_vs_run2 06:00:00 \
    --run_a ${R1} \
    --run_b ${R2} \
    ${REBEN_DATA}

submit reben_AS1_full_run1_vs_run3 06:00:00 \
    --run_a ${R1} \
    --run_b ${R3} \
    ${REBEN_DATA}

submit reben_AS1_full_run2_vs_run3 06:00:00 \
    --run_a ${R2} \
    --run_b ${R3} \
    ${REBEN_DATA}

# same checkpoint, two heads: encoder identical, only the decoder heads are interpolated
submit reben_AS2_full_run1_h0_vs_h1 06:00:00 \
    --run_a ${R_AS2} \
    --run_b ${R_AS2} \
    --head_a 0 \
    --head_b 1 \
    ${REBEN_DATA}

# ---- FBP (frozen encoder, decoder only; full dataset, so no --max_images; ~12 min on a P100) ----
submit fbp_AS1_full_run1_vs_run2 02:00:00 \
    --pipeline fbp \
    --run_a ${F1} \
    --run_b ${F2}

submit fbp_AS1_full_run1_vs_run3 02:00:00 \
    --pipeline fbp \
    --run_a ${F1} \
    --run_b ${F3}

submit fbp_AS1_full_run2_vs_run3 02:00:00 \
    --pipeline fbp \
    --run_a ${F2} \
    --run_b ${F3}

submit fbp_AS2_full_run1_h0_vs_h1 02:00:00 \
    --pipeline fbp \
    --run_a ${F_AS2_1} \
    --run_b ${F_AS2_1} \
    --head_a 0 \
    --head_b 1

submit fbp_AS2_full_run1_vs_run2_h0 02:00:00 \
    --pipeline fbp \
    --run_a ${F_AS2_1} \
    --run_b ${F_AS2_2} \
    --head_a 0 \
    --head_b 0
