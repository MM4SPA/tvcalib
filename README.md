# TVCalib: Camera Calibration for Sports Field Registration in Soccer

This repository contains the implementation for TVCalib and its application and evaluation on the [SoccerNet Calibration Challenge](https://www.soccer-net.org/tasks/calibration).

### Table of Contents
1. [Camera Calibration Part](#Camera-Calibration-Part)
2. [Segment Localization Part](#Segment-Localization-Part)
3. [Post-processing, Visualization, and Evaluation](#Post-processing,-Visualization,-and-Evaluation)
5. [Resource Files](#Resource-Files)
6. [Reproduce Paper Results](#Reproduce-Results)


## Camera Calibration Part

To predict all individual camera parameters from the results of the segment localization run: `optimize.py`:
Arguments:
- `--hparams`: Path to config filem, e.g., `configs/val_main_center_gt.json` (see details below)
- `--output_dir`: Default `./experiments`; extends with (hparams.stem), e.g., `val_main_center_gt`
- `--log_per_step`: If given, log inforamtion like loss during each optimization step for each sample
- `--visualize_results`: Plot results (3d projecteded pitch with overlayed input image and points) for each batch
- `--device`: `cuda` or `cpu`, default ``cuda`

### Config File (hparams) - Example

```json
{
    "temporal_dim": 1,
    "batch_dim": 256,
    "sigma_scale": 1.96,
    "object3d": "SoccerPitchLineCircleSegments",
    "dataset": {
        "file_match_info": "/nfs/data/soccernet/calibration/test/match_info.json",
        "extremities_annotations": "data/extremities/gt/valid",
        "extremities_prefix": "extremities_",
        "num_points_on_line_segments": 4,
        "num_points_on_circle_segments": 4,
        "filter_cam_type": null,
        "remove_invalid": true
    },
    "lens_distortion": false,
    "image_width": 960,
    "image_height": 540,
    "optim_steps": 1000
}
```
- `num_points_on_line_segments` and `num_points_on_circle_segments`: Randomly samples points from provided extremities. If the number of given points is lower, the input is padded with zeros.
- `split_circle_central`: If set to `true`, the central sircle is divided into left and right part using a heuristic.
- `remove_invalid`: Only relevant if `temporal_dim` > 1 as it removes samples from the dataset which can not fulfill the required number of images per stadium and camera type.
- `extremities_annotations` refers to a folder comprising `<extremities_prefix><image_id>.json` files with following information. 

### Output

The folder `output_dir_prefix` contains at least the predicted camera parameters and additional information (e.g., loss, other meta infomration like stadium, image ids) for each batch (`batch_{idx}.pt`) 



## Segment Localization Part
Note, that annotations from a segmen localization model or ground-truth annotations can serve es input for the calibration module (`extremities_annotations`) where each sample (`image_id.json`) is represented by: 
```json
{
     "semantic_class_name_1" : [{'x': x1,'y': y1}, {'x': x2,'y': y2}],
     "semantic_class_name_2": [{'x': x3,'y': y3}, {'x': x4,'y': y4}]
      ...
}
```

<hr>

## Post-processing, Visualization, and Evaluation

`python -m evaluation.eval_and_summarize` performs post-processing of the output, filtering to remove invalid samples (self-verification), evaluation, and aggregates the results.
Arguments:
- `--dir_results`: Directory containing the output of *TVCalib* (folder with `batch_{idx}.pt`)
- `--dataset_dir`: Corresponding directory containing the ground truth annotations, e.g., `/nfs/data/soccernet/calibration/valid`
- `--taus`: `Thesholds to ignore probably invalid samples during evaluation, default=[np.inf, 0.03, 0.025, 0.02, 0.017]
- `--plots_only`: If provided, generate `per_sample_output.json` and some plots only; do not perform evaluation.

### Post-processing

The raw (per-batch) output is flattend (per-sample) and stored in `per_sample_output.json`, where each line represents one sample, for example:

```json
{
    "image_ids":"1.jpg",
    "time_s":0.0433,
    "mask_lines":[[[false,false,false],[true,true,false],[true,true,true],...]],
    "mask_circles":[[[false,false,false,false,false,false,false,false],[true,true,true,true,true,true,true,true], ...]],
    "loss_ndc_lines_distances_raw":[[[0.0,0.0,0.0],[0.010469364,0.0019166876,0.0],[0.0086851967,0.0050904173,0.0100935074],...]],
    "loss_ndc_lines":0.0047949357,
    "loss_ndc_circles_distances_raw":[[[0.0,0.0,0.0,0.0],[0.0100957388,0.0031360432,0.0003662109,0.0027729045],...]],
    "loss_ndc_circles":0.0039352393,
    "loss_ndc_total":0.008730175,
    "loss_ndc_lines_distances_max":0.010469364,
    "loss_ndc_circles_distances_max":0.0100957388,
    "loss_ndc_total_max":0.010469364,
    "league":"Fifa WorldCup",
    "season":"2014",
    "match":"None",
    "date":"None",
    "pan_degrees":-17.578212738,
    "tilt_degrees":80.8040008545,
    "roll_degrees":-0.1310900003,
    "position_meters":[-7.7424321175,57.8480377197,-11.1651697159],
    "aov_radian":0.4283054769,
    "aov_degrees":24.540096283,
    "x_focal_length":2942.6950683594,
    "y_focal_length":2942.6950683594,
    "principal_point":[640.0,360.0],
    "radial_distortion":[0.0,0.0,0.0,0.0,0.0,0.0],
    "tangential_distortion":[0.0,0.0],
    "thin_prism_distortion":[0.0,0.0,0.0,0.0],
    "stadium":"None",
}
```

### Evaluation
See [Evaluation](https://github.com/SoccerNet/sn-calibration#evaluation-1) for details on the evaluation metric.

`eval_and_summarize` integrates the metric calculation from the official evaluation script, but with modified input pipeline and result aggregation. 

Modifications to the [original version](https://github.com/SoccerNet/sn-calibration#evaluation-1):
- Replaced the folder of `camera.json` files with `per_sample_output.json`
- As we can filter out probably invalid samples (based on the total loss) evaluation can automatically performed for a couple of provided thresholds
- The official evaluation script needs to be executed multiple times (to evaluate different evaluation thresholds (5px, 10px, 20px)). Now, all thresholds are evaluated in one call.
- trivial way for multiprocessing
- summarize raw outputs in a `pd.DataFrame`

<hr>


## Resource Files

- To download the SN-Calib dataset, follow the instructions in https://github.com/SoccerNet/sn-calibration
- [Camera Type Annotations](https://tvcalib.open-develop.org/cameras.zip): Our manually annotated and predicted camera types for each sample
- [Segment Localization](https://tvcalib.open-develop.org/extremities.zip): The output of our retrained model (source code and inference script will be available soon)

## Reproduce Results
- To fully reproduce the results on SN-Calib, run the scripts in `scripts/` for the respective dataset
- [Results](https://tvcalib.open-develop.org/experiments.zip): Pre-computed results: Results can be inspected via `evaluation/summarize_result_folder_{valid_test}`.
- Results on WC14: available soon
- Homography Decomposition from already computed homographies (state-of-the-art approaches, manually annotated): available soon