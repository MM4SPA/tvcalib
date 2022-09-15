#!/bin/zsh

TAUS=(0.019)
OUTPUT_DIR="experiments/lens_distortion/sncalib-valid-center"
DATASET_DIR="data/datasets/sncalib-valid"
WIDTH=960
HEIGHT=540

for CONFIG_FILE_PREFIX in "extrem-pred" "extrem-gt";do
python -m tvcalib.optimize --output_dir $OUTPUT_DIR --hparams configs/lens_distortion/sncalib-valid-center/$CONFIG_FILE_PREFIX.json
python -m evaluation.eval_projection --filter_gt_camera_type "Main camera center" --dir_dataset $DATASET_DIR --width $WIDTH --height $HEIGHT --per_sample_output $OUTPUT_DIR/$CONFIG_FILE_PREFIX/per_sample_output.json --project_from Camera --taus $TAUS --evaluate_3d
done

for CONFIG_FILE_PREFIX in "extrem-pred-ld" "extrem-gt-ld";do
python -m tvcalib.optimize --output_dir $OUTPUT_DIR --hparams configs/lens_distortion/sncalib-valid-center/$CONFIG_FILE_PREFIX.json
python -m evaluation.eval_projection --distort --filter_gt_camera_type "Main camera center" --dir_dataset $DATASET_DIR --width $WIDTH --height $HEIGHT --per_sample_output $OUTPUT_DIR/$CONFIG_FILE_PREFIX/per_sample_output.json --project_from Camera --taus $TAUS --evaluate_3d 
done


# TAUS=(inf)
OUTPUT_DIR="experiments/lens_distortion/wc14-test"
DATASET_DIR="data/datasets/wc14-test"
WIDTH=1280
HEIGHT=720

CONFIG_FILE_PREFIX="extrem-gt-ld"
python -m tvcalib.optimize --output_dir $OUTPUT_DIR --hparams configs/lens_distortion/wc14-test/$CONFIG_FILE_PREFIX.json --visualize_results
python -m evaluation.eval_projection --distort --dir_dataset $DATASET_DIR --width $WIDTH --height $HEIGHT --per_sample_output $OUTPUT_DIR/$CONFIG_FILE_PREFIX/per_sample_output.json --project_from Camera --taus $TAUS --evaluate_3d 

CONFIG_FILE_PREFIX="extrem-gt"
python -m tvcalib.optimize --output_dir $OUTPUT_DIR --hparams configs/lens_distortion/wc14-test/$CONFIG_FILE_PREFIX.json
python -m evaluation.eval_projection --dir_dataset $DATASET_DIR --width $WIDTH --height $HEIGHT --per_sample_output $OUTPUT_DIR/$CONFIG_FILE_PREFIX/per_sample_output.json --project_from Camera --taus $TAUS --evaluate_3d
