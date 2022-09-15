#!/bin/zsh
SAVE_DIR="experiments/wc14-test"
DATASET_DIR="data/datasets/wc14-test"
WIDTH=1280
HEIGHT=720

# ################################## Manually Annotated Homography Matrices H ##################################
python scripts/write_per_sample_output_from_wc14_homography.py --input_dir data/datasets/wc14-test --output_dir $SAVE_DIR/h_manual-extrem_gt
python -m evaluation.eval_projection --per_sample_output $SAVE_DIR/h_manual-extrem_gt/per_sample_output.json --project_from HDecomp Homography --evaluate_3d --evaluate_2d --dir_dataset $DATASET_DIR --width $WIDTH --height $HEIGHT
