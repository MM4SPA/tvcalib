import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from tvcalib.utils.objects_3d import SoccerPitchSN

np.set_printoptions(precision=4, suppress=True)

from SoccerNet.Evaluation.utils_calibration import (
    get_polylines,
    scale_points,
    evaluate_camera_prediction,
    mirror_labels,
)
from SoccerNet.Evaluation.utils_calibration import Camera


class WC14Homography:
    def __init__(self, H=None) -> None:

        T = np.eye(3)
        T[0, -1] = -115 / 2
        T[1, -1] = -74 / 2
        yard2meter = 0.9144
        S = np.eye(3)
        S[0, 0] = yard2meter
        S[1, 1] = yard2meter

        self.H = S @ (T @ H)  # transform WC14 coordinate system to SN
        self.H_inv = np.linalg.inv(self.H)
        self.image_width = 1280.0
        self.image_height = 720.0

    def project_point(self, p):
        # projets a template point (x, y, z) to image
        # assume that z == 0.0
        projected = self.H_inv @ p
        projected /= projected[2]
        return projected  # [x, y, 1]

    def draw_colorful_pitch(self, image, palette):
        raise NotImplementedError


# def get_polylines(
#     cam, width, height, sampling_factor=0.2, ignore_goal=False, force_z_to_zero=False
# ):
#     """
#     Given a set of camera parameters, this function adapts the camera to the desired image resolution and then
#     projects the 3D points belonging to the terrain model in order to give a dictionary associating the classes
#     observed and the points projected in the image.

#     :param camera_annotation: camera parameters in their json/dictionary format
#     :param width: image width for evaluation
#     :param height: image height for evaluation
#     :return: a dictionary with keys corresponding to a class observed in the image ( a line of the 3D model whose
#     projection falls in the image) and values are then the list of 2D projected points.
#     """

#     if isinstance(cam, Camera):
#         if cam.image_width != width:
#             cam.scale_resolution(width / cam.image_width)
#     field = SoccerPitchSN()
#     projections = dict()
#     sides = [
#         np.array([1, 0, 0]),
#         np.array([1, 0, -width + 1]),
#         np.array([0, 1, 0]),
#         np.array([0, 1, -height + 1]),
#     ]
#     for key, points in field.sample_field_points(sampling_factor).items():
#         if ignore_goal:
#             if "Goal" in key:  # TODO can be removed if ignored in annotated_line_segments
#                 continue
#         projections_list = []
#         in_img = False
#         prev_proj = np.zeros(3)
#         for i, point in enumerate(points):
#             if force_z_to_zero:
#                 # need to set z=0 when evaluating homography!
#                 point[-1] = 1.0
#             ext = cam.project_point(point)
#             if ext[2] < 1e-5:
#                 # point at infinity or behind camera
#                 continue
#             if 0 <= ext[0] < width and 0 <= ext[1] < height:

#                 if not in_img and i > 0:

#                     line = np.cross(ext, prev_proj)
#                     in_img_intersections = []
#                     dist_to_ext = []
#                     for side in sides:
#                         intersection = np.cross(line, side)
#                         intersection /= intersection[2]
#                         if 0 <= intersection[0] < width and 0 <= intersection[1] < height:
#                             in_img_intersections.append(intersection)
#                             dist_to_ext.append(np.sqrt(np.sum(np.square(intersection - ext))))
#                     if in_img_intersections:
#                         intersection = in_img_intersections[np.argmin(dist_to_ext)]

#                         projections_list.append({"x": intersection[0], "y": intersection[1]})

#                 projections_list.append({"x": ext[0], "y": ext[1]})
#                 in_img = True
#             elif in_img:
#                 # first point out
#                 line = np.cross(ext, prev_proj)

#                 in_img_intersections = []
#                 dist_to_ext = []
#                 for side in sides:
#                     intersection = np.cross(line, side)
#                     intersection /= intersection[2]
#                     if 0 <= intersection[0] < width and 0 <= intersection[1] < height:
#                         in_img_intersections.append(intersection)
#                         dist_to_ext.append(np.sqrt(np.sum(np.square(intersection - ext))))
#                 if in_img_intersections:
#                     intersection = in_img_intersections[np.argmin(dist_to_ext)]

#                     projections_list.append({"x": intersection[0], "y": intersection[1]})
#                 in_img = False
#             prev_proj = ext
#         if len(projections_list):
#             projections[key] = projections_list
#     return projections


def eval_df(
    x,
    annotation_files,
    resolution_width,
    resolution_height,
    ignore_goal=False,
    sampling_factor_lines=0.2,
):
    df_subset, eval_threshold, subset_idx = x
    print(eval_threshold)

    accuracies = []
    precisions = []
    recalls = []
    dict_errors = {}  # TODO: need to log?
    per_class_confusion_dict = {}  # TODO: need to log?
    missed, total_frames = 0, 0
    for i, annotation_file in enumerate(annotation_files):
        if i % 500 == 0:
            print("subset_idx", subset_idx, i, "/", len(annotation_files))
        frame_index = annotation_file.name.split(".")[0]
        # annotation_file = dataset_dir / annotation_file # for WC14 eval comment this line

        total_frames += 1

        if frame_index not in df_subset.index:
            missed += 1
            continue

        with open(annotation_file, "r") as f:
            line_annotations = json.load(f)

        predictions = df_subset.loc[frame_index]

        line_annotations = scale_points(line_annotations, resolution_width, resolution_height)

        img_groundtruth = line_annotations

        # cam = Camera()
        # cam.from_json_parameters(predictions)

        # img_prediction = get_polylines(
        #     cam,
        #     resolution_width,
        #     resolution_height,
        #     sampling_factor=sampling_factor_lines,
        #     ignore_goal=ignore_goal,
        # )

        img_prediction = get_polylines(
            predictions,
            resolution_width,
            resolution_height,
            sampling_factor=sampling_factor_lines,
        )

        confusion1, per_class_conf1, reproj_errors1 = evaluate_camera_prediction(
            img_prediction, img_groundtruth, eval_threshold
        )

        confusion2, per_class_conf2, reproj_errors2 = evaluate_camera_prediction(
            img_prediction, mirror_labels(img_groundtruth), eval_threshold
        )

        accuracy1, accuracy2 = 0.0, 0.0
        if confusion1.sum() > 0:
            accuracy1 = confusion1[0, 0] / confusion1.sum()

        if confusion2.sum() > 0:
            accuracy2 = confusion2[0, 0] / confusion2.sum()

        if accuracy1 > accuracy2:
            accuracy = accuracy1
            confusion = confusion1
            # per_class_conf = per_class_conf1
            # reproj_errors = reproj_errors1
        else:
            accuracy = accuracy2
            confusion = confusion2
            # per_class_conf = per_class_conf2
            # reproj_errors = reproj_errors2

        accuracies.append(accuracy)
        if confusion[0, :].sum() > 0:
            precision = confusion[0, 0] / (confusion[0, :].sum())
            precisions.append(precision)
        if (confusion[0, 0] + confusion[1, 0]) > 0:
            recall = confusion[0, 0] / (confusion[0, 0] + confusion[1, 0])
            recalls.append(recall)

    mRecall = np.mean(recalls)
    sRecall = np.std(recalls)
    medianRecall = np.median(recalls)

    mPrecision = np.mean(precisions)
    sPrecision = np.std(precisions)
    medianPrecision = np.median(precisions)

    mAccuracy = np.mean(accuracies)
    sAccuracy = np.std(accuracies)
    medianAccuracy = np.median(accuracies)

    return {
        "total completeness rate": f"{(total_frames - missed) / total_frames} ({total_frames - missed}/{total_frames})",
        "accuracy mean": mAccuracy,
        "recall mean": mRecall,
        "precision mean": mPrecision,
        "accuracy median": medianAccuracy,
        "recall median": medianRecall,
        "precision median": medianPrecision,
        "accuracy std": sAccuracy,
        "recall std": sRecall,
        "precision std": sPrecision,
    }


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluation camera calibration task")

    parser.add_argument(
        "-s",
        "--soccernet",
        default="./annotations",
        type=str,
        help="Path to the SoccerNet-V3 dataset folder",
    )
    parser.add_argument(
        "-p",
        "--prediction",
        default="./results_bis",
        required=False,
        type=str,
        help="Path to the prediction folder",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        default=20,
        required=False,
        type=int,
        help="Accuracy threshold in pixels",
    )
    parser.add_argument(
        "--split",
        required=False,
        type=str,
        default="test",
        help="Select the split of data",
    )
    parser.add_argument(
        "--resolution_width",
        required=False,
        type=int,
        default=960,
        help="width resolution of the images",
    )
    parser.add_argument(
        "--resolution_height",
        required=False,
        type=int,
        default=540,
        help="height resolution of the images",
    )
    parser.add_argument("--use_wc14_homography", action="store_true")
    parser.add_argument("--ignore_non_plane", action="store_true")
    parser.add_argument("--sampling_factor_lines", type=float, default=0.9)
    args = parser.parse_args()

    accuracies = []
    precisions = []
    recalls = []
    dict_errors = {}
    per_class_confusion_dict = {}

    dataset_dir = os.path.join(args.soccernet, args.split)
    if not os.path.exists(dataset_dir):
        print("Invalid dataset path !")
        exit(-1)

    if args.use_wc14_homography and args.ignore_non_plane == False:
        raise RuntimeError("if use_wc14_homography -> ignore_non_plane should be set to True")

    annotation_files = [
        f for f in os.listdir(dataset_dir) if ".json" in f and f != "match_info.json"
    ]
    # ISSUE all *.json files are considered as GT --> also match_info.json, etc.

    assert len(annotation_files) > 0

    missed, total_frames = 0, 0
    with tqdm(enumerate(annotation_files), total=len(annotation_files), ncols=160) as t:
        for i, annotation_file in t:
            frame_index = annotation_file.split(".")[0]
            annotation_file = os.path.join(args.soccernet, args.split, annotation_file)

            if args.use_wc14_homography == True:
                prediction_file = os.path.join(args.prediction, f"{frame_index}.homographyMatrix")
            else:
                prediction_file = os.path.join(
                    args.prediction, args.split, f"camera_{frame_index}.json"
                )

            total_frames += 1

            if not os.path.exists(prediction_file):
                missed += 1

                continue

            with open(annotation_file, "r") as f:
                line_annotations = json.load(f)

            # exclude all segments where z=! 0; e.g., when using a homography
            if args.ignore_non_plane:
                line_annotations = {k: v for k, v in line_annotations.items() if "Goal" not in k}

            if args.use_wc14_homography == True:
                cam = WC14Homography(np.loadtxt(prediction_file))
            else:
                with open(prediction_file, "r") as f:
                    predictions = json.load(f)
                cam = Camera()
                cam.from_json_parameters(predictions)

            line_annotations = scale_points(
                line_annotations, args.resolution_width, args.resolution_height
            )

            image_path = os.path.join(args.soccernet, args.split, f"{frame_index}.jpg")

            img_groundtruth = line_annotations

            img_prediction = get_polylines(
                cam,
                args.resolution_width,
                args.resolution_height,
                sampling_factor=args.sampling_factor_lines,
                ignore_goal=args.ignore_non_plane,
                force_z_to_zero=args.use_wc14_homography,
            )

            confusion1, per_class_conf1, reproj_errors1 = evaluate_camera_prediction(
                img_prediction, img_groundtruth, args.threshold
            )

            confusion2, per_class_conf2, reproj_errors2 = evaluate_camera_prediction(
                img_prediction, mirror_labels(img_groundtruth), args.threshold
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

    print(
        f" On SoccerNet {args.split} set, completeness rate of : {(total_frames - missed) / total_frames}"
    )
    mRecall = np.mean(recalls)
    sRecall = np.std(recalls)
    medianRecall = np.median(recalls)
    print(
        f" On SoccerNet {args.split} set, recall mean value : {mRecall * 100:2.2f}% with standard deviation of {sRecall * 100:2.2f}% and median of {medianRecall * 100:2.2f}%"
    )

    mPrecision = np.mean(precisions)
    sPrecision = np.std(precisions)
    medianPrecision = np.median(precisions)
    print(
        f" On SoccerNet {args.split} set, precision mean value : {mPrecision * 100:2.2f}% with standard deviation of {sPrecision * 100:2.2f}% and median of {medianPrecision * 100:2.2f}%"
    )

    mAccuracy = np.mean(accuracies)
    sAccuracy = np.std(accuracies)
    medianAccuracy = np.median(accuracies)
    print(
        f" On SoccerNet {args.split} set, accuracy mean value :  {mAccuracy * 100:2.2f}% with standard deviation of {sAccuracy * 100:2.2f}% and median of {medianAccuracy * 100:2.2f}%"
    )

    print()

    for line_class, confusion_mat in per_class_confusion_dict.items():
        class_accuracy = confusion_mat[0, 0] / confusion_mat.sum()
        class_recall = confusion_mat[0, 0] / (confusion_mat[0, 0] + confusion_mat[1, 0])
        class_precision = confusion_mat[0, 0] / (confusion_mat[0, 0] + confusion_mat[0, 1])
        print(
            f"For class {line_class}, accuracy of {class_accuracy * 100:2.2f}%, precision of {class_precision * 100:2.2f}%  and recall of {class_recall * 100:2.2f}%"
        )

        for k, v in dict_errors.items():
            fig, ax1 = plt.subplots(figsize=(11, 8))
            ax1.hist(v, bins=30, range=(0, 60))
            ax1.set_title(k)
            ax1.set_xlabel("Errors in pixel")
            os.makedirs(f"./results/", exist_ok=True)
            plt.savefig(f"./results/{k}_reprojection_error.png")
            plt.close(fig)
