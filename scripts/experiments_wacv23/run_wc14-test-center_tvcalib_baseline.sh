#!/bin/zsh

SAVE_DIR="experiments/wc14-test"
DATASET_DIR="data/datasets/wc14-test"
WIDTH=1280
HEIGHT=720


# ################################## TVCALIB ##################################
TAUS=(inf 0.019)

CONFIG_FILE_PREFIX="extrem-gt"
python -m tvcalib.optimize --output_dir $SAVE_DIR --hparams configs/wc14-test/$CONFIG_FILE_PREFIX.json
python -m evaluation.eval_projection --dir_dataset $DATASET_DIR --width $WIDTH --height $HEIGHT --per_sample_output $SAVE_DIR/$CONFIG_FILE_PREFIX/per_sample_output.json --project_from Camera --taus $TAUS --evaluate_3d --evaluate_2d

CONFIG_FILE_PREFIX="extrem-pred"
python -m tvcalib.optimize --output_dir $SAVE_DIR --hparams configs/wc14-test/$CONFIG_FILE_PREFIX.json
python -m evaluation.eval_projection --dir_dataset $DATASET_DIR --width $WIDTH --height $HEIGHT --per_sample_output $SAVE_DIR/$CONFIG_FILE_PREFIX/per_sample_output.json --project_from Camera --taus $TAUS --evaluate_3d --evaluate_2d


# ################################## Baseline: Homography from Line Correspondences ##################################

# GT Annotations
PREFIX="Hline-extrem-gt"
python scripts/baseline_homography_from_line_correspondences.py --extremities $DATASET_DIR --extremities_prefix "" --output_dir $SAVE_DIR/$PREFIX --width $WIDTH --height $HEIGHT
python -m evaluation.eval_projection --per_sample_output $SAVE_DIR/$PREFIX/per_sample_output.json --project_from Homography HDecomp --evaluate_2d --evaluate_3d --dir_dataset $DATASET_DIR --width $WIDTH --height $HEIGHT 

# Retrained segmentation model with original point selection -> two points are enough
PREFIX="Hline-extrem-pred"
python scripts/baseline_homography_from_line_correspondences.py --extremities data/segment_localization/np2_nc8_r4_md30/wc14-test --output_dir $SAVE_DIR/$PREFIX --width $WIDTH --height $HEIGHT
python -m evaluation.eval_projection --per_sample_output $SAVE_DIR/$PREFIX/per_sample_output.json --project_from Homography HDecomp --evaluate_2d --evaluate_3d --dir_dataset $DATASET_DIR --width $WIDTH --height $HEIGHT
