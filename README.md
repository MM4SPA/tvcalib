# TVCalib: Camera Calibration for Sports Field Registration in Soccer

<div align="center">

[![Project](https://img.shields.io/badge/mm4spa.github.io-TVCalib-538135.svg?style=for-the-badge)](https://mm4spa.github.io/tvcalib/)
[![Conference](https://img.shields.io/badge/WACV-2023-6b8bc7.svg?style=for-the-badge)](https://arxiv.org/abs/2207.11709)
[![arXiv](https://img.shields.io/badge/arXiv-2207.11709-b31b1b.svg?style=for-the-badge)](https://arxiv.org/abs/2207.11709)

</div>

## Contents
- [Inference / Demo](#inference)
- [Reproduce Paper Results](#reproduce-paper-results)
    - [Datasets](#datasets)
    - [Evaluation](#evaluation)
- [Requirments](#requirements)

<hr>  


# Inference
`inference.ipynb`: Given a bunch of images, semantic segmentation, point selection, estimation of camera parameters, and visualization is applied. The pretrained segmentation model can be downloaded here:
```bash
mkdir data/segment_localization 
wget https://tib.eu/cloud/s/x68XnTcZmsY4Jpg/download/train_59.pt -O data/segment_localization/train_59.pt
```

# [Visualize Results](tvcalib/visualize_per_sample_output.ipynb)


# Reproduce Paper Results

We provide scripts (`scripts/experiments_wacv23`) to reproduce the provided results of the paper for the baseline and *TVCalib*.

```bash

# SN segmentation model & retrained model
scripts/experiments_wacv23/run_segmentation.sh
# choice of self-verification parameter
scripts/experiments_wacv23/run_sncalib-valid-all-tau_tvcalib.sh
# TVCalib & baseline
scripts/experiments_wacv23/run_wc14-test-center_tvcalib_baseline.sh
scripts/experiments_wacv23/run_sncalib-test-center_tvcalib_baseline.sh
scripts/experiments_wacv23/run_lens_distortion_tvcalib.sh

scripts/experiments_wacv23/run_wc14-test-center_manual.sh

# +++ further ablation studies
scripts/experiments_wacv23/run_sncalib-test-all_tvcalib.sh

# table 1
python scripts/experiments_wacv23/tex/generate_table_sncalib-center.py
# table 2, 3
python scripts/experiments_wacv23/tex/generate_table_wc14-center.py
# table appendix: lens distortion
python scripts/experiments_wacv23/tex/generate_table_lens_distortion.py
# figure 2: segment reprojection loss
python scripts/experiments_wacv23/figures/visualize_ndc_losses_multiple_datasets.py
# figure 3: sn-calib-test (main left, center, right)
python scripts/experiments_wacv23/figures/summarize_results_sncalib-test-all.py
# evaluate projection error
python -m scripts.experiments_wacv23.tex.prepare_iou_results
```

## Evaluation

### Segment Reprojection Error
See [https://github.com/SoccerNet/sn-calibration](https://github.com/SoccerNet/sn-calibration#evaluation-1) for details on the evaluation metric.

`python -m evaluation.eval_projection`

Arguments:
- `--dir_dataset`: Path to ground-truth annotations, i.e., a folder with `<image_id>.json`
- `--filter_gt_camera_type <str>`: Evaluate on a subset according to the available camera types in `<dir_dataset>/match_info_cam_gt.json`
- `--per_sample_output`: File path to `per_sample_output_json`
- `--width <int> --height <int>`: Source image with and height in pixel, respectively
- `--project_from`: `[Camera, Homography, HDecomp]` while `Camera` requires individual camera parameters, `Homography` and `HDecomp` a respective homography matrix. Multiple values possible.
- `--evaluate_3d`: If set, evaluates the 3D calibration performance (utilizes the 3D pitch model)
- `--evaluate_2d`: If set, evaluates the 2D calibration performance from a provided homography (utilizes the 2D pitch model)
- `--distort`: Evaluate with provided lens distortion parameters. Default: ignored
- `--taus` (optional): Relevant for TVCalib only: Self-verification from loss. Example, `--taus inf 0.017`
- `--zeta` (optional): Relevant for `project_from=HDecomp`.


### Projection Error via Intersection over Union (Part):
See `python -m scripts.experiments_wacv23.tex.prepare_iou_results`.

## Datasets

Expected structure for default arguments: 
```
./
├── data
│   └── datasets
│       └── wc14-test/match_info_cam_gt.json
│       └── sncalib-train/match_info_cam_gt.json
│       └── sncalib-valid/match_info_cam_gt.json
│       └── sncalib-test/match_info_cam_gt.json
```
Download and preparation:

### SoccerNet-Calibration-V3:

```python
from SoccerNet.Downloader import SoccerNetDownloader
mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory="</nfs/data/soccernet>")
mySoccerNetDownloader.downloadDataTask(task="calibration", split=["train","valid","test"])
```

Already downloaded? May consider to create a soft link for each subset: 
```bash
ln -s /nfs/data/soccernet/calibration/valid data/datasets/sncalib-valid
ln -s /nfs/data/soccernet/calibration/test data/datasets/sncalib-test
ln -s /nfs/data/soccernet/calibration/train data/datasets/sncalib-train
```

Camera type annotations
```bash
# move annotation file to respective dataset directory
wget https://tib.eu/cloud/s/483Bqf78dDMcx2H/download/test_match_info_cam_gt.json -O sncalib-test/match_info_cam_gt.json
wget https://tib.eu/cloud/s/WdSqM3WbyKQ36pm/download/val_match_info_cam_gt.json -O sncalib-valid/match_info_cam_gt.json

```

### WorldCup 2014 (WC14):

```bash
mkdir -p data/datasets/wc14-test && cd data/datasets/wc14-test/
# Images and provided homography matrices from test split
wget https://nhoma.github.io/data/soccer_data.tar.gz
tar -zxvf soccer_data.tar.gz raw/test --strip-components 2
# Our additional segment annotations
wget https://tib.eu/cloud/s/Jz4x2KsjinEEkwQ/download/wc14-test-additional_annotations_wacv23_theiner.tar -O wc14-test-additional_annotations_wacv23_theiner.tar
tar xvf wc14-test-additional_annotations_wacv23_theiner.tar
```

# Requirements

### [Conda Environment](https://docs.conda.io/):

```bash
conda env create -f environment.yml
conda activate tvcalib
```
Depending on your hardware, consider to have a look on https://pytorch.org/ for CPU-only installation or other CUDA versions.
