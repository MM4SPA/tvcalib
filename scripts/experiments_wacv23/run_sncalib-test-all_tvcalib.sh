#!/bin/bash

DATASET_DIR="data/datasets/sncalib-test"
OUTPUT_DIR="experiments/sncalib-test"
TAUS="0.025 0.023 0.021 0.019 0.017 0.015 0.013 0.011 0.01 0.009"


# TVCalib (argmin) - GT extremities vs Pred extrimities
for CONFIG_FILE_PREFIX in "extrem-pred" "extrem-gt";do
python -m tvcalib.optimize --hparams configs/sncalib-test/$CONFIG_FILE_PREFIX.json --output_dir $OUTPUT_DIR/argmin/center --overwrite_init_cam_distr "Main camera center"
python -m tvcalib.optimize --hparams configs/sncalib-test/$CONFIG_FILE_PREFIX.json --output_dir $OUTPUT_DIR/argmin/left --overwrite_init_cam_distr "Main camera left"
python -m tvcalib.optimize --hparams configs/sncalib-test/$CONFIG_FILE_PREFIX.json --output_dir $OUTPUT_DIR/argmin/right --overwrite_init_cam_distr "Main camera right"
python -m tvcalib.fuse_argmin --result_dir_base $OUTPUT_DIR/argmin --subset_glob "?/${CONFIG_FILE_PREFIX}" --subsets center left right
python -m evaluation.eval_projection --per_sample_output $OUTPUT_DIR/argmin/argmin/$CONFIG_FILE_PREFIX/per_sample_output.json --dir_dataset $DATASET_DIR --taus $TAUS --dir_dataset $DATASET_DIR --width 960 --height 540 --project_from Camera --evaluate_3d
done

# TVCalib (stacked) - GT extremities vs Pred extrimities - Cam GT
for EXTREM in "pred" "gt";do
python -m tvcalib.optimize --hparams configs/sncalib-test/left_extrem-${EXTREM}_cam-gt.json --output_dir $OUTPUT_DIR/stacked
python -m tvcalib.optimize --hparams configs/sncalib-test/right_extrem-${EXTREM}_cam-gt.json --output_dir $OUTPUT_DIR/stacked
python -m tvcalib.optimize --hparams configs/sncalib-test/center_extrem-${EXTREM}_cam-gt.json --output_dir $OUTPUT_DIR/stacked
python -m tvcalib.fuse_stack --result_dir_base $OUTPUT_DIR/stacked --subset_glob "?_extrem-${EXTREM}_cam-gt" --subsets center left right
python -m evaluation.eval_projection --per_sample_output $OUTPUT_DIR/stacked/stacked_extrem-${EXTREM}_cam-gt/per_sample_output.json --taus $TAUS --dir_dataset $DATASET_DIR --width 960 --height 540 --project_from Camera --evaluate_3d
done

# TVCalib (center) - GT extrimities - Cam no filtering vs. cam center
CONFIG_FILE_PREFIX="extrem-gt"
python -m tvcalib.optimize --output_dir $OUTPUT_DIR --hparams configs/sncalib-test/$CONFIG_FILE_PREFIX.json --overwrite_init_cam_distr "Main tribune"
python -m evaluation.eval_projection --per_sample_output $OUTPUT_DIR/$CONFIG_FILE_PREFIX/per_sample_output.json --taus $TAUS --dir_dataset $DATASET_DIR --width 960 --height 540 --project_from Camera --evaluate_3d

CONFIG_FILE_PREFIX="center_extrem-gt_cam-gt"
python -m tvcalib.optimize --output_dir $OUTPUT_DIR --hparams configs/sncalib-test/$CONFIG_FILE_PREFIX.json
python -m evaluation.eval_projection --per_sample_output $OUTPUT_DIR/$CONFIG_FILE_PREFIX/per_sample_output.json --taus $TAUS --dir_dataset $DATASET_DIR --width 960 --height 540 --project_from Camera --evaluate_3d
