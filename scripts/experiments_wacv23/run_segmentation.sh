#!/bin/zsh

OUTPUT_DIR="data/segment_localization"

DATASET_PATH="data/datasets"
CKPT="data/segment_localization/train_59.pt"
num_points_circles=8


num_points_lines=4
python -m sn_segmentation.src.custom_extremities -s $DATASET_PATH -p $OUTPUT_DIR --checkpoint $CKPT --num_points_lines $num_points_lines --num_points_circles $num_points_circles --split wc14-test
python -m sn_segmentation.src.custom_extremities -s $DATASET_PATH -p $OUTPUT_DIR --checkpoint $CKPT --num_points_lines $num_points_lines --num_points_circles $num_points_circles --split sncalib-valid
python -m sn_segmentation.src.custom_extremities -s $DATASET_PATH -p $OUTPUT_DIR --checkpoint $CKPT --num_points_lines $num_points_lines --num_points_circles $num_points_circles --split sncalib-test


# for DLT from line segments: baseline -> best results for two major points
num_points_lines=2
python -m sn_segmentation.src.custom_extremities -s $DATASET_PATH -p $OUTPUT_DIR --checkpoint $CKPT --num_points_lines $num_points_lines --num_points_circles $num_points_circles --split wc14-test
python -m sn_segmentation.src.custom_extremities -s $DATASET_PATH -p $OUTPUT_DIR --checkpoint $CKPT --num_points_lines $num_points_lines --num_points_circles $num_points_circles --split sncalib-valid
python -m sn_segmentation.src.custom_extremities -s $DATASET_PATH -p $OUTPUT_DIR --checkpoint $CKPT --num_points_lines $num_points_lines --num_points_circles $num_points_circles --split sncalib-test


# ResNet-50 original
python -m sn_segmentation.src.baseline_extremities --checkpoint_dir sn_segmentation/resources -s $DATASET_PATH -p $OUTPUT_DIR/sn-baseline --split wc14-test --masks true
python -m sn_segmentation.src.baseline_extremities --checkpoint_dir sn_segmentation/resources -s $DATASET_PATH -p $OUTPUT_DIR/sn-baseline --split sncalib-test
# ResNet-101 retrained
python -m sn_segmentation.src.custom_extremities -s $DATASET_PATH -p $OUTPUT_DIR --checkpoint $CKPT --split wc14-test --num_points_lines 2 --pp_maxdists 40
python -m sn_segmentation.src.custom_extremities -s $DATASET_PATH -p $OUTPUT_DIR --checkpoint $CKPT --split sncalib-test --num_points_lines 2 --pp_maxdists 40
