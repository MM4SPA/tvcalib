#!/bin/zsh

# Optimal choice of tau for main camera center
OUTPUT_DIR="experiments/sncalib-valid-cam_center-tau"
DATASET_DIR="data/datasets/sncalib-valid"
WIDTH=960
HEIGHT=540
TAUS=(inf 0.025 0.024 0.023 0.022 0.021 0.02 0.019 0.018 0.017 0.016 0.015 0.014 0.013)

CONFIG_FILE_PREFIX="extrem-pred"
python -m tvcalib.optimize --output_dir $OUTPUT_DIR --hparams configs/sncalib-valid-center/$CONFIG_FILE_PREFIX.json
python -m evaluation.eval_projection --filter_gt_camera_type "Main camera center" --dir_dataset $DATASET_DIR --width $WIDTH --height $HEIGHT --per_sample_output $OUTPUT_DIR/$CONFIG_FILE_PREFIX/per_sample_output.json --project_from Camera --taus $TAUS --evaluate_3d