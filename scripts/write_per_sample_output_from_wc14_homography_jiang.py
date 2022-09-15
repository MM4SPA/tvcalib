from pathlib import Path
import numpy as np
import pandas as pd
from argparse import ArgumentParser

args = ArgumentParser()
args.add_argument(
    "--input",
    type=Path,
    required=True,
)
args.add_argument("--output_dir", type=Path, required=True)
args = args.parse_args()


# assume already created results
# https://github.com/vcg-uvic/sportsfield_release


class ConvertHomographyJiang_to_WC14GT:
    def __init__(self, h_jiang) -> None:
        # Jiang et al. scale S_templ @ (H_wc14 @ S_frame)
        # i.e. 1) frame [+-0.5] -> [720, 1280] 2) [+- 115, 74] -> [+-0.5] and 3) normalize s.t. last entry == 1
        S_frame = np.array([[1280.0, 0, 640], [0, 720, 360], [0, 0, 1]])
        S_templ = np.array(
            [[0.008695654, 0, -0.5], [0, 0.013513514, -0.5], [0, 0, 1]]
        )  # ~ inverse(S_frame)

        # inverse function
        self.H = np.linalg.inv(S_templ) @ h_jiang @ np.linalg.inv(S_frame)
        self.H = self.H / self.H[-1, -1]

    def get_homography(self):
        return self.H


class ConvertHomography_WC14GT_to_SN:
    def __init__(self, H=None) -> None:
        # /!\ H maps from image to world (template)
        T = np.eye(3)
        T[0, -1] = -115 / 2
        T[1, -1] = -74 / 2
        yard2meter = 0.9144
        S = np.eye(3)
        S[0, 0] = yard2meter
        S[1, 1] = yard2meter

        self.H = S @ (T @ H)  # transform WC14 coordinate system to SN
        # self.H_inv = np.linalg.inv(self.H)
        self.image_width = 1280.0
        self.image_height = 720.0

    def get_homography(self):
        return self.H


h_file = args.input
df_per_sample = pd.read_json(h_file)
df_per_sample["homography"] = df_per_sample["homography"].apply(lambda h: np.array(h[0]))
df_per_sample["homography"] = df_per_sample["homography"].apply(
    lambda h: ConvertHomography_WC14GT_to_SN(ConvertHomographyJiang_to_WC14GT(h).get_homography())
    .get_homography()
    .tolist()
)

args.output_dir.mkdir(exist_ok=True, parents=True)
df_per_sample.to_json(args.output_dir / "per_sample_output.json", orient="records", lines=True)
