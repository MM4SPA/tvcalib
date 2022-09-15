import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np

from SoccerNet.Evaluation.utils_calibration import SoccerPitch

"""Writes per_sample_output.json for Baseline Homography from Line Correspondences.

"""


def normalization_transform(points):
    """
    Computes the similarity transform such that the list of points is centered around (0,0) and that its distance to the
    center is sqrt(2).
    :param points: point cloud that we wish to normalize
    :return: the affine transformation matrix
    """
    center = np.mean(points, axis=0)

    d = 0.0
    nelems = 0
    for p in points:
        nelems += 1
        x = p[0] - center[0]
        y = p[1] - center[1]
        di = np.sqrt(x**2 + y**2)
        d += (di - d) / nelems

    if d <= 0.0:
        s = 1.0
    else:
        s = np.sqrt(2) / d
    T = np.zeros((3, 3))
    T[0, 0] = s
    T[0, 2] = -s * center[0]
    T[1, 1] = s
    T[1, 2] = -s * center[1]
    T[2, 2] = 1
    return T


def estimate_homography_from_line_correspondences(lines, T1=np.eye(3), T2=np.eye(3)):
    """
    Given lines correspondences, computes the homography that maps best the two set of lines.
    :param lines: list of pair of 2D lines matches.
    :param T1: Similarity transform to normalize the elements of the source reference system
    :param T2: Similarity transform to normalize the elements of the target reference system
    :return: boolean to indicate success or failure of the estimation, homography
    """
    homography = np.eye(3)
    A = np.zeros((len(lines) * 2, 9))

    for i, line_pair in enumerate(lines):
        src_line = np.transpose(np.linalg.inv(T1)) @ line_pair[0]
        target_line = np.transpose(np.linalg.inv(T2)) @ line_pair[1]
        u = src_line[0]
        v = src_line[1]
        w = src_line[2]

        x = target_line[0]
        y = target_line[1]
        z = target_line[2]

        A[2 * i, 0] = 0
        A[2 * i, 1] = x * w
        A[2 * i, 2] = -x * v
        A[2 * i, 3] = 0
        A[2 * i, 4] = y * w
        A[2 * i, 5] = -v * y
        A[2 * i, 6] = 0
        A[2 * i, 7] = z * w
        A[2 * i, 8] = -v * z

        A[2 * i + 1, 0] = x * w
        A[2 * i + 1, 1] = 0
        A[2 * i + 1, 2] = -x * u
        A[2 * i + 1, 3] = y * w
        A[2 * i + 1, 4] = 0
        A[2 * i + 1, 5] = -u * y
        A[2 * i + 1, 6] = z * w
        A[2 * i + 1, 7] = 0
        A[2 * i + 1, 8] = -u * z

    try:
        u, s, vh = np.linalg.svd(A)
    except np.linalg.LinAlgError:
        return False, homography
    v = np.eye(3)
    has_positive_singular_value = False
    for i in range(s.shape[0] - 1, -2, -1):
        v = np.reshape(vh[i], (3, 3))

        if s[i] > 0:
            has_positive_singular_value = True
            break

    if not has_positive_singular_value:
        return False, homography

    homography = np.reshape(v, (3, 3))
    homography = np.linalg.inv(T2) @ homography @ T1
    homography /= homography[2, 2]

    return True, homography


parser = argparse.ArgumentParser()
parser.add_argument(
    "--extremities",
    required=True,
    type=Path,
    help="Path to the prediction folder",
)
parser.add_argument(
    "--output_dir",
    required=True,
    type=Path,
)
parser.add_argument(
    "--extremities_prefix",
    type=str,
    default="extremities_",
)
parser.add_argument(
    "--width",
    required=True,
    type=int,
)
parser.add_argument(
    "--height",
    required=True,
    type=int,
)
args = parser.parse_args()

field = SoccerPitch()  # 3D model


files_extremities = [f for f in args.extremities.glob(f"{args.extremities_prefix}*.json")]
files_extremities = [f for f in files_extremities if "match_info" not in f.name]
assert len(files_extremities) > 0


records = []
for file_prediction in files_extremities:
    frame_index = file_prediction.stem[len(args.extremities_prefix) :]

    with open(file_prediction, "r") as fr:
        predictions = json.load(fr)

    line_matches = []
    potential_3d_2d_matches = {}
    src_pts = []
    success = False
    for k, v in predictions.items():
        if k == "Circle central" or "unknown" in k:  # What about 'Circle left/right'?
            continue
        P3D1 = field.line_extremities_keys[k][0]
        P3D2 = field.line_extremities_keys[k][1]
        p1 = np.array([v[0]["x"] * args.width, v[0]["y"] * args.height, 1.0])
        p2 = np.array([v[1]["x"] * args.width, v[1]["y"] * args.height, 1.0])
        src_pts.extend([p1, p2])
        if P3D1 in potential_3d_2d_matches.keys():
            potential_3d_2d_matches[P3D1].extend([p1, p2])
        else:
            potential_3d_2d_matches[P3D1] = [p1, p2]
        if P3D2 in potential_3d_2d_matches.keys():
            potential_3d_2d_matches[P3D2].extend([p1, p2])
        else:
            potential_3d_2d_matches[P3D2] = [p1, p2]

        start = (int(p1[0]), int(p1[1]))
        end = (int(p2[0]), int(p2[1]))

        line = np.cross(p1, p2)
        if np.isnan(np.sum(line)) or np.isinf(np.sum(line)):
            continue
        line_pitch = field.get_2d_homogeneous_line(k)
        if line_pitch is not None:
            line_matches.append((line_pitch, line))

    if len(line_matches) >= 4:
        target_pts = [field.point_dict[k][:2] for k in potential_3d_2d_matches.keys()]
        T1 = normalization_transform(target_pts)
        T2 = normalization_transform(src_pts)
        success, homography = estimate_homography_from_line_correspondences(line_matches, T1, T2)
        if success:
            homography = np.linalg.inv(homography)

            records.append({"image_ids": frame_index + ".jpg", "homography": homography.tolist()})

df_per_sample = pd.DataFrame.from_records(records)
args.output_dir.mkdir(exist_ok=True, parents=True)
df_per_sample.to_json(args.output_dir / "per_sample_output.json", orient="records", lines=True)

print(df_per_sample)
print("num success:", len(df_per_sample.index), "/", len(files_extremities))
