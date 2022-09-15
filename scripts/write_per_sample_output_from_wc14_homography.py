from pathlib import Path
import numpy as np
import pandas as pd
from argparse import ArgumentParser

args = ArgumentParser()
args.add_argument(
    "--input_dir",
    type=Path,
    help="Path to original manually annotated homography matrices *.homographyMatrix",
    required=True,
)
args.add_argument("--output_dir", type=Path, required=True)
args = args.parse_args()


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


h_dir = args.input_dir
h_files = [f for f in h_dir.glob("*.homographyMatrix")]
print(f"found {len(h_files)} files matching *.homographyMatrix")
assert len(h_files) == 186

records = []
for h_file in h_files:

    image_id = h_file.stem

    h: np.ndarray = np.loadtxt(h_file)  # 3x3
    # convert to SN-Calib world coordinate system
    h = ConvertHomography_WC14GT_to_SN(h).get_homography()
    records.append(
        {
            "image_ids": str(image_id) + ".jpg",
            "homography": h.tolist(),
            "image_width": 1280,
            "image_height": 720,
        }
    )

df_per_sample = pd.DataFrame.from_records(records)

args.output_dir.mkdir(exist_ok=True, parents=True)
df_per_sample.to_json(args.output_dir / "per_sample_output.json", orient="records", lines=True)
