from argparse import ArgumentParser
from functools import partial
import json
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd
from SoccerNet.Evaluation.utils_calibration import (
    get_polylines,
    scale_points,
    evaluate_camera_prediction,
    mirror_labels,
)


def evaluate(x, gt_jsons, width=960, height=540):
    df_subset, threshold = x

    accuracies = []
    precisions = []
    recalls = []
    dict_errors = {}
    per_class_confusion_dict = {}
    total_frames = 0
    missed = 0

    for i, gt_json in enumerate(gt_jsons):
        if i % 500 == 0:
            print(i + 1, "/", len(gt_jsons), "eval_threshold=", threshold)
        frame_index = gt_json.name.split(".")[0]

        total_frames += 1
        if frame_index not in df_subset.index:
            missed += 1
            continue

        df_subset.loc[frame_index]
        with open(gt_json, "r") as fr:
            gt = json.load(fr)

        line_annotations = scale_points(gt, width, height)

        img_groundtruth = line_annotations

        prediction = df_subset.loc[frame_index]

        img_prediction = get_polylines(prediction, width, height, sampling_factor=0.9)

        confusion1, per_class_conf1, reproj_errors1 = evaluate_camera_prediction(
            img_prediction, img_groundtruth, threshold
        )

        confusion2, per_class_conf2, reproj_errors2 = evaluate_camera_prediction(
            img_prediction, mirror_labels(img_groundtruth), threshold
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

        for line_class, errors in reproj_errors.items():
            if line_class in dict_errors.keys():
                dict_errors[line_class].extend(errors)
            else:
                dict_errors[line_class] = errors

        for line_class, confusion_mat in per_class_conf.items():
            if line_class in per_class_confusion_dict.keys():
                per_class_confusion_dict[line_class] += confusion_mat
            else:
                per_class_confusion_dict[line_class] = confusion_mat

    # print(len(gt_jsons), missed)
    # print((len(gt_jsons) - missed) / len(gt_jsons))

    results = {}
    results["completeness"] = (total_frames - missed) / total_frames
    results["meanRecall"] = np.mean(recalls)
    results["meanPrecision"] = np.mean(precisions)
    results["meanAccuracies"] = np.mean(accuracies)

    for line_class, confusion_mat in per_class_confusion_dict.items():
        class_accuracy = confusion_mat[0, 0] / confusion_mat.sum()
        class_recall = confusion_mat[0, 0] / (confusion_mat[0, 0] + confusion_mat[1, 0])
        class_precision = confusion_mat[0, 0] / (confusion_mat[0, 0] + confusion_mat[0, 1])
        results[f"{line_class}Precision"] = class_precision
        results[f"{line_class}Recall"] = class_recall
        results[f"{line_class}Accuracy"] = class_accuracy
    return results


if __name__ == "__main__":

    eval_thresholds = [5, 10, 20]  # in px

    args = ArgumentParser()
    args.add_argument("--dir_results", type=Path)
    args.add_argument("--dataset_dir", type=Path)
    args.add_argument("--plots_only", action="store_true")
    args.add_argument("--resolution_width", type=int, default=960)
    args.add_argument("--resolution_height", type=int, default=540)
    args.add_argument(
        "--taus",
        nargs="+",
        help="e.g. --taus inf 0.05 0.04 0.03 0.025 0.02 0.017",
        default=["inf", "0.05", "0.04", "0.03", "0.025", "0.02", "0.017", "0.015"],
    )
    args.add_argument("--sampling_factor_lines", type=float, default=0.9)
    args.add_argument("--ignore_goal", action="store_true")
    args.add_argument("--from_per_sample_output", action="store_true")
    args.add_argument("--nworker", type=int, default=16)
    args = args.parse_args()
    dir_results = args.dir_results
    dataset_dir = args.dataset_dir
    plots_only = args.plots_only
    taus = [float(x) for x in args.taus]
    resolution_width = args.resolution_width
    resolution_height = args.resolution_height

    # read predictions
    fout = dir_results / "per_sample_output.json"
    if not fout.exists():
        raise FileNotFoundError(fout)
    print("Read from already created", dir_results / "per_sample_output.json")
    df = pd.read_json(fout, orient="records", lines=True)
    df["image_id"] = df["image_ids"].apply(lambda l: l.split(".jpg")[0])
    df.set_index("image_id", inplace=True)

    # read ground truth annotations
    gt_jsons = [f for f in dataset_dir.glob("*.json")]
    gt_jsons = [f for f in gt_jsons if "match_info" not in f.name]
    assert len(gt_jsons) > 0

    dataset_size_unfiltered = len(df.index)
    result_records = []
    df_subsets = []
    eval_thresholds_flat = []
    for tau in taus:
        for eval_threshold in eval_thresholds:
            df_subset = df.copy()
            df_subset = df_subset.dropna(subset=["aov_radian"])
            df_subset = df.loc[df["loss_ndc_total"] <= tau]
            # df_subset = df_subset.sample(n=50) # debug
            size_subset_tau_filter = len(df_subset)
            df_subsets.append(df_subset)
            eval_thresholds_flat.append(eval_threshold)
            result_records.append(
                {
                    "tau": tau,
                    "threshold": eval_threshold,
                    "completeness (subset)": f"{len(df_subset.index) / dataset_size_unfiltered} ({len(df_subset.index)}/{dataset_size_unfiltered})",
                }
            )

    nruns = len(taus) * len(eval_thresholds)
    fn_evaluate = partial(evaluate, gt_jsons=gt_jsons)
    with Pool(args.nworker) as p:
        summary_records = p.map(fn_evaluate, list(zip(df_subsets, eval_thresholds_flat)))

    i = 0
    for tau in taus:
        for eval_threshold in eval_thresholds:
            result_records[i].update(summary_records[i])
            i += 1

    df_summary = pd.DataFrame.from_records(result_records)
    df_summary.sort_values(["tau", "threshold"], ascending=False, inplace=True)
    print(df_summary)

    fout = dir_results / "df_summary_full.json"
    print(fout)
    df_summary.to_json(fout, orient="records", lines=True)
    # from long to wide format and column selection
    col_select = [
        "completeness",
        "completeness (subset)",
        "meanAccuracies",
    ]
    index_select = ["completeness", "completeness (subset)"]
    df_summary_wide = df_summary[["tau", "threshold"] + col_select].pivot(
        index=["tau"] + index_select, columns="threshold"
    )
    df_summary_wide.sort_index(ascending=False, inplace=True)
    print(df_summary_wide)

    #%%
    fout = dir_results / "df_summary_wide_acc_mean.txt"
    print(fout)
    df_summary_wide.to_string(fout, col_space=4)
