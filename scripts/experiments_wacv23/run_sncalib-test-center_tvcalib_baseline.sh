#!/bin/zsh

SAVE_DIR="experiments/sncalib-test-cam_center"
DATASET_DIR="data/datasets/sncalib-test"
WIDTH=960
HEIGHT=540


# ################################## TVCALIB ##################################
TAUS=(inf 0.019)

CONFIG_FILE_PREFIX="extrem-gt"
python -m tvcalib.optimize --output_dir $SAVE_DIR --hparams configs/sncalib-test-center/$CONFIG_FILE_PREFIX.json
python -m evaluation.eval_projection --filter_gt_camera_type "Main camera center" --dir_dataset $DATASET_DIR --width $WIDTH --height $HEIGHT --per_sample_output $SAVE_DIR/$CONFIG_FILE_PREFIX/per_sample_output.json --project_from Camera --taus $TAUS --evaluate_3d --evaluate_2d

CONFIG_FILE_PREFIX="extrem-pred"
python -m tvcalib.optimize --output_dir $SAVE_DIR --hparams configs/sncalib-test-center/$CONFIG_FILE_PREFIX.json
python -m evaluation.eval_projection --filter_gt_camera_type "Main camera center" --dir_dataset $DATASET_DIR --width $WIDTH --height $HEIGHT --per_sample_output $SAVE_DIR/$CONFIG_FILE_PREFIX/per_sample_output.json --project_from Camera --taus $TAUS --evaluate_3d --evaluate_2d


# ################################## Baseline: Homography from Line Correspondences ##################################

# GT Annotations
python scripts/baseline_homography_from_line_correspondences.py --extremities $DATASET_DIR --extremities_prefix "" --output_dir $SAVE_DIR/Hline-extrem-gt --width $WIDTH --height $HEIGHT
python -m evaluation.eval_projection --filter_gt_camera_type "Main camera center" --per_sample_output $SAVE_DIR/Hline-extrem-gt/per_sample_output.json --project_from Homography HDecomp --evaluate_2d --evaluate_3d --dir_dataset $DATASET_DIR --width $WIDTH --height $HEIGHT 

# Retrained segmentation model with original point selection -> two points are enough
python scripts/baseline_homography_from_line_correspondences.py --extremities data/segment_localization/np2_nc8_r4_md30/sncalib-test --output_dir $SAVE_DIR/Hline-extrem-pred --width $WIDTH --height $HEIGHT
python -m evaluation.eval_projection --filter_gt_camera_type "Main camera center" --per_sample_output $SAVE_DIR/Hline-extrem-pred/per_sample_output.json --project_from Homography HDecomp --evaluate_2d --evaluate_3d --dir_dataset $DATASET_DIR --width $WIDTH --height $HEIGHT
