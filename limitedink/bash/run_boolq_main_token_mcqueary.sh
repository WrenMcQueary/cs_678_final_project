#!/usr/bin/env bash
source /home/wren/miniconda3/etc/profile.d/conda.sh
conda activate cs_678_final_project_environment_ubuntu
set -x;
set -e;
REPO_PATH="/mnt/c/Users/User/Documents/ClassNotes/QuickNotes/PythonProjects/cs_678_final_project"
export PYTHONPATH=$REPO_PATH

k=0.3;
rand="1234";
SAVE_DIR="$REPO_PATH/checkpoints/boolq/distilbert/token_rationale/length_level_$k/seed_$rand";
mkdir -p $SAVE_DIR
LOG_DIR="$SAVE_DIR/train.log";

CUDA_VISIBLE_DEVICES=0 python ../main.py --data_dir "$REPO_PATH/data/boolq" --save_dir $SAVE_DIR --configs "$REPO_PATH/limitedink/params/boolq_config_token.json" --length $k --seed $rand > $LOG_DIR

