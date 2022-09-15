#%%
import pickle
import numpy as np
from pathlib import Path
import pandas as pd
import h5py
from argparse import ArgumentParser


class ConvertHomography_Chen_to_SN:
    def __init__(self, H=None) -> None:
        T = np.eye(3)
        T[0, -1] = -105 / 2
        T[1, -1] = -68 / 2
        self.H = T @ H

        self.H = np.array([[1.0, 0, 0], [0, -1, 0], [0, 0, 1]]) @ self.H

    def get_homography(self):
        return self.H


class ConvertSN720p2SN540p:
    def __init__(self, H=None) -> None:
        s_f = 960 / 1280
        S = np.array([[s_f, 0, 0], [0, s_f, 0], [0, 0, 1.0]])

        self.h = H @ np.linalg.inv(S)
        self.h = self.h / self.h[-1, -1]

    def get_homography(self):
        return self.h


args = ArgumentParser()
args.add_argument("--result_dir", type=Path)
args.add_argument("--output", type=str)
args.add_argument("--image_ids", type=str, default="seg_image_paths.pkl")
args = args.parse_args()

h_file = args.result_dir / args.output
print(h_file)
image_ids_file = args.result_dir / args.image_ids
with h5py.File(h_file) as h5_dataset:
    hs_refined = h5_dataset["h_refined_ext"][()]  # (N, 3, 4)
    hs_refined = hs_refined[:, :, [0, 1, 3]]
    hs_init = h5_dataset["h_ext"][()]  # (N, 3, 4)
    hs_init = hs_init[:, :, [0, 1, 3]]

with open(image_ids_file, "rb") as fr:
    image_paths = pickle.load(fr)


records = []
for image_id, h, h_init in zip(image_paths, hs_refined, hs_init):
    h = h / h[-1, -1]  # normalize
    h = np.linalg.inv(h)
    h = h / h[-1, -1]  # normalize
    h = ConvertSN720p2SN540p(ConvertHomography_Chen_to_SN(h).get_homography()).get_homography()

    h_init = h_init / h_init[-1, -1]  # normalize
    h_init = np.linalg.inv(h_init)
    h_init = h_init / h_init[-1, -1]  # normalize
    h_init = ConvertSN720p2SN540p(
        ConvertHomography_Chen_to_SN(h_init).get_homography()
    ).get_homography()

    records.append(
        {
            "image_ids": image_id,
            "homography": h.tolist(),
            "homography_init": h_init.tolist(),
        }
    )


df_per_sample = pd.DataFrame.from_records(records)
df_per_sample.to_json(args.result_dir / "per_sample_output.json", orient="records", lines=True)
