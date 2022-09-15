import itertools
import json
from multiprocessing import Pool
from pathlib import Path
from argparse import ArgumentParser, Namespace
from functools import partial
from typing import List

import numpy as np
import pandas as pd
from SoccerNet.Evaluation.utils_calibration import (
    scale_points,
    evaluate_camera_prediction,
    mirror_labels,
)

from evaluation.utils import get_polylines, SoccerPitch2D, compound_score


def evaluate(
    x: Namespace,
    gt_jsons: list,
    sampling_factor: float,
    distort: bool,
    width: float,
    height: float,
    wrap_delta: float,
):

    accuracies = []
    precisions = []
    recalls = []
    dict_errors = {}
    per_class_confusion_dict = {}
    frames_passed = 0

    for i, gt_json in enumerate(gt_jsons):
        if i % 1000 == 0:
            print(
                f"{i}/{len(gt_jsons)}, {x.project_from=} {x.evaluate_planar=} t={x.eval_threshold}px, {x.tau=}, {x.zeta=}"
            )
        frame_index = gt_json.name.split(".")[0]  # /frame_index.json -> frame_index

        if frame_index not in x.df_subset.index:  # expected but not available in predicted
            continue  # missed+=1

        with open(gt_json) as fr:
            gt = json.load(fr)  # i.e. annotated points

        img_groundtruth = scale_points(gt, width, height)

        # either individual camera parameters or homography matrix
        prediction = x.df_subset.loc[frame_index]

        img_prediction = get_polylines(
            prediction,
            width,
            height,
            sampling_factor,
            x.project_from,
            x.evaluate_planar,
            distort,
            x.zeta,
            wrap_delta,
        )
        if img_prediction is None:
            continue

        frames_passed += 1

        if x.evaluate_planar:  # filter GT segments where z != 0
            img_groundtruth = {
                k: v
                for k, v in img_groundtruth.items()
                if (k in SoccerPitch2D.lines_classes) or ("Circle" in k)
            }

        confusion1, per_class_conf1, reproj_errors1 = evaluate_camera_prediction(
            img_prediction, img_groundtruth, x.eval_threshold
        )

        confusion2, per_class_conf2, reproj_errors2 = evaluate_camera_prediction(
            img_prediction, mirror_labels(img_groundtruth), x.eval_threshold
        )
        accuracy1, accuracy2 = 0.0, 0.0
        if confusion1.sum() > 0:
            accuracy1 = confusion1[0, 0] / confusion1.sum()

        if confusion2.sum() > 0:
            accuracy2 = confusion2[0, 0] / confusion2.sum()

        if accuracy1 > accuracy2:
            accuracy = accuracy1
            confusion = confusion1
            per_class_conf = per_class_conf1
            reproj_errors = reproj_errors1
        else:
            accuracy = accuracy2
            confusion = confusion2
            per_class_conf = per_class_conf2
            reproj_errors = reproj_errors2

        accuracies.append(accuracy)
        if confusion[0, :].sum() > 0:
            precision = confusion[0, 0] / (confusion[0, :].sum())
            precisions.append(precision)
        if (confusion[0, 0] + confusion[1, 0]) > 0:
            recall = confusion[0, 0] / (confusion[0, 0] + confusion[1, 0])
            recalls.append(recall)

        for segment_key, errors in reproj_errors.items():
            if segment_key in dict_errors.keys():
                dict_errors[segment_key].extend(errors)
            else:
                dict_errors[segment_key] = errors

        for segment_key, confusion_mat in per_class_conf.items():
            if segment_key in per_class_confusion_dict.keys():
                per_class_confusion_dict[segment_key] += confusion_mat
            else:
                per_class_confusion_dict[segment_key] = confusion_mat

    results = {}
    results["completeness_raw"] = frames_passed
    results["meanRecall"] = np.mean(recalls)
    results["meanPrecision"] = np.mean(precisions)
    results["meanAccuracies"] = np.mean(accuracies)

    for segment_key, confusion_mat in per_class_confusion_dict.items():
        class_accuracy = confusion_mat[0, 0] / confusion_mat.sum()
        class_recall = confusion_mat[0, 0] / (confusion_mat[0, 0] + confusion_mat[1, 0])
        class_precision = confusion_mat[0, 0] / (confusion_mat[0, 0] + confusion_mat[0, 1])

        results[f"{segment_key} Precision"] = class_precision
        results[f"{segment_key} Recall"] = class_recall
        results[f"{segment_key} Accuracy"] = class_accuracy
    return results


def main(args):
    EVALUATE_PLANAR = []
    if args.evaluate_3d == True:
        EVALUATE_PLANAR.append(False)
    if args.evaluate_2d == True:
        EVALUATE_PLANAR.append(True)
    # if no distortion coeff are estimated -> no need to perform extra computations
    DISTORT = True if args.distort else False
    ZETA = [float(zeta) for zeta in args.zeta]

    per_sample_output = args.per_sample_output
    if not per_sample_output.exists():
        raise FileNotFoundError(per_sample_output)
    dir_results = per_sample_output.parent

    TAUS = [None]
    if args.taus:
        TAUS = args.taus
    PROJECT_FROM = args.project_from

    gt_jsons: List[Path] = None  # List of files where the filename is the sample_id
    gt_jsons = [f for f in args.dir_dataset.glob("*.json")]
    gt_jsons = [f for f in gt_jsons if "match_info" not in f.name]
    dataset_size_total = len(gt_jsons)
    dataset_size_subset = dataset_size_total
    if args.filter_gt_camera_type:
        assert (args.dir_dataset / "match_info_cam_gt.json").exists()
        df_filter = pd.read_json(args.dir_dataset / "match_info_cam_gt.json").T
        df_filter["image_id"] = [s.split(".jpg")[0] for s in df_filter.index]
        df_filter = df_filter.loc[df_filter.camera == args.filter_gt_camera_type]
        print(df_filter)
        allowed_gt = set(df_filter.image_id.tolist())
        gt_jsons = [f for f in gt_jsons if f.stem in allowed_gt]
        dataset_size_subset = len(gt_jsons)
    assert len(gt_jsons) > 0

    # index col -> sample_id: columns: Camera parameters (defined in SNCamera) or "homography"
    # Note: assert that H is created or converted to the used world coordinate system
    df_subset: pd.DataFrame = None

    df = pd.read_json(per_sample_output, orient="records", lines=True)
    df["image_id"] = df["image_ids"].apply(lambda l: l.split(".jpg")[0])
    df = df.drop_duplicates(subset="image_id")
    df.set_index("image_id", inplace=True)

    fn_eval = partial(
        evaluate,
        gt_jsons=gt_jsons,
        width=args.width,
        height=args.height,
        sampling_factor=0.9,
        wrap_delta=WRAP_DELTA,
        distort=DISTORT,
    )

    # cartesian product from all variants
    cartesian_product = []
    for x in itertools.product(*[PROJECT_FROM, TAUS, EVAL_THRESHOLDS, EVALUATE_PLANAR, ZETA]):
        project_from, tau, eval_threshold, evaluate_planar, zeta = x
        if project_from == "Homography" and evaluate_planar == False:
            continue
        if project_from == "Homography" and isinstance(zeta, float):
            zeta = None
        if project_from != "HDecomp":
            zeta = None
        cartesian_product.append((project_from, tau, eval_threshold, evaluate_planar, zeta))

    print("Variants to evaluate:")
    eval_params_flat = []
    for x in set(cartesian_product):
        print(x)
        project_from, tau, eval_threshold, evaluate_planar, zeta = x
        df_subset = df.copy()
        if tau is not None and tau != "inf":
            tau = float(tau)
            df_subset = df.copy().loc[df["loss_ndc_total"] <= tau]
            df_subset = df_subset.dropna(axis=0, subset="loss_ndc_total")
        else:
            pass  # use full dataset
        eval_params_flat.append(
            Namespace(
                **{
                    "project_from": project_from,
                    "dataset_size": dataset_size_total,
                    "dataset_size_subset": dataset_size_subset,
                    "df_subset": df_subset,
                    "tau": tau,
                    "evaluate_planar": evaluate_planar,
                    "eval_threshold": eval_threshold,
                    "zeta": zeta,
                    "wrap_delta": WRAP_DELTA if project_from == "HDecomp" else None,
                }
            )
        )
    fout = dir_results / "results_raw.json"
    # if not fout.exists():
    with Pool(args.nworkers) as p:
        result_records = p.map(fn_eval, eval_params_flat)

    for i in range(len(eval_params_flat)):
        result_records[i].update(vars(eval_params_flat[i]))
        del result_records[i]["df_subset"]

    df = pd.DataFrame.from_records(result_records)

    print(fout)
    df.to_json(fout, orient="records", lines=True)

    #################################################################
    df_summary = pd.read_json(fout, orient="records", lines=True)
    df_summary["completeness_subset"] = (
        df_summary["completeness_raw"] / df_summary["dataset_size_subset"]
    )
    df_summary["completeness"] = df_summary["completeness_raw"] / df_summary["dataset_size"]

    # from long to wide format and column selection
    col_select = [
        "completeness",
        "completeness_subset",
        "meanAccuracies",
    ]
    index_select = ["completeness", "completeness_subset"]
    df_summary_wide = df_summary[
        ["tau", "eval_threshold", "project_from", "evaluate_planar", "zeta"] + col_select
    ].pivot(
        index=["project_from", "evaluate_planar", "tau", "zeta"] + index_select,
        columns="eval_threshold",
    )

    df_summary_wide.sort_index(ascending=False, inplace=True)
    df_summary_wide.reset_index(inplace=True, col_level=1)
    df_summary_wide.columns = [
        f"{k}@{v}" if k == "meanAccuracies" else v for (k, v) in df_summary_wide.columns
    ]

    df_summary_wide["compound_score"] = compound_score(
        df_summary_wide["completeness"],
        *[df_summary_wide[f"meanAccuracies@{t}"] for t in EVAL_THRESHOLDS],
    )
    df_summary_wide["compound_score_subset"] = compound_score(
        df_summary_wide["completeness"],
        *[df_summary_wide[f"meanAccuracies@{t}"] for t in EVAL_THRESHOLDS],
    )

    # all metrics to percentage
    for k in [
        "compound_score",
        "compound_score_subset",
        "completeness",
        "completeness_subset",
        *[f"meanAccuracies@{t}" for t in EVAL_THRESHOLDS],
    ]:
        df_summary_wide[k] = df_summary_wide[k] * 100.0

    print(df_summary_wide)
    fout_prefix = "eval_calibration_summary"
    fout = dir_results / f"{fout_prefix}.json"
    print(fout)
    df_summary_wide.to_json(fout, orient="records", lines=True)


if __name__ == "__main__":

    EVAL_THRESHOLDS = [20, 10, 5]
    SAMPLING_FACTOR = 0.9
    WRAP_DELTA = 0.1

    args = ArgumentParser()
    args.add_argument("--dir_dataset", type=Path, required=True)
    args.add_argument("--width", type=int, required=True)
    args.add_argument("--height", type=int, required=True)
    # output and evaluation
    args.add_argument("--per_sample_output", type=Path, required=True)
    args.add_argument("--project_from", nargs="+")  # ["HDecomp", "Homography", "Camera"]
    args.add_argument("--evaluate_3d", action="store_true")
    args.add_argument("--evaluate_2d", action="store_true")
    args.add_argument(
        "--taus",
        nargs="+",
        help="Optional and only relevant for TVCalib e.g. --taus inf 0.02, 0.017",
    )
    args.add_argument("--distort", action="store_true", help="eval with distortion parameters")
    args.add_argument(
        "--zeta",
        nargs="+",
        help="Optional and only relevant for project_from=HDecomp. e.g. --zeta 100.0 300.0, 1000.0",
        default=[100.0],
    )
    args.add_argument("--filter_gt_camera_type", type=str)
    args.add_argument("--nworkers", type=int, default=16)
    args = args.parse_args()
    main(args)
