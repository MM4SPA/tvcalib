from distutils.errors import UnknownFileError
from scipy.io import loadmat
import pickle
import numpy as np
import h5py
from pathlib import Path
import pandas as pd
from argparse import ArgumentParser

args = ArgumentParser()
args.add_argument("--result_dir", type=Path, required=True)
args.add_argument("--image_ids", type=Path, default="seg_image_paths.pkl")
args.add_argument("--input", type=Path, required=True)
args = args.parse_args()


class ConvertHomography_Chen_to_SN:
    def __init__(self, H=None) -> None:
        T = np.eye(3)
        T[0, -1] = -105 / 2
        T[1, -1] = -68 / 2
        self.H = T @ H

        self.H = np.array([[1.0, 0, 0], [0, -1, 0], [0, 0, 1]]) @ self.H

    def get_homography(self):
        return self.H

print(args.input)
if args.input.suffix == ".mat":
    data = loadmat(args.input)
    data = data["homography"]  # (186, 3, 3)
elif args.input.suffix == ".h5":
    with h5py.File(args.input) as data:
        print(data.keys())
        data = data["h_refined_ext"][()]  # (N, 3, 4)
        data = data[:, :, [0, 1, 3]]
        # hs_init = data["h_ext"][()]  # (N, 3, 4)
        # hs_init = hs_init[:, :, [0, 1, 3]]
else:
    raise UnknownFileError

with open(args.image_ids, "rb") as fr:
    image_paths = pickle.load(fr)


records = []
for image_id, h in zip(image_paths, data):
    h = h / h[-1, -1]  # normalize
    h = np.linalg.inv(h)
    h = h / h[-1, -1]  # normalize
    h = ConvertHomography_Chen_to_SN(h).get_homography()

    records.append(
        {
            "image_ids": image_id,
            "homography": h.tolist(),
            "image_width": 1280,
            "image_height": 720,
        }
    )


df_per_sample = pd.DataFrame.from_records(records)
args.result_dir.mkdir(exist_ok=True, parents=True)
df_per_sample.to_json(args.result_dir / "per_sample_output.json", orient="records", lines=True)
