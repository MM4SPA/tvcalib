from argparse import ArgumentParser
import json
from pathlib import Path
import pandas as pd
import zipfile
from io import StringIO

"""Generates output required for online evaluation server, i.e., a `.zip` including `json` files 
named "camera_{frame_index}.json" each containing a dictionary with the camera parameters.
"""

args = ArgumentParser()
args.add_argument(
    "--per_sample_output",
    type=Path,
    required=True,
    help="Per sample output file (**/per_sample_output.json)",
)
args.add_argument("--tau", type=float, default=100000)
args = args.parse_args()

file_per_sample_output = args.per_sample_output
tau = args.tau

FILTER_KEY = "loss_ndc_total"
COL_SELECT = [
    "pan_degrees",
    "tilt_degrees",
    "roll_degrees",
    "position_meters",
    "x_focal_length",
    "y_focal_length",
    "principal_point",
    "radial_distortion",
    "tangential_distortion",
    "thin_prism_distortion",
]

if not file_per_sample_output.exists():
    raise FileNotFoundError(file_per_sample_output)

df = pd.read_json(file_per_sample_output, orient="records", lines=True)
df.set_index("image_ids", inplace=True, drop=False)
df = df.dropna(subset=["aov_radian"])
df = df.loc[df[FILTER_KEY] <= tau]  # remove probably errornous samples
df = df[COL_SELECT]

file_zip_out = (
    file_per_sample_output.parent / f"{file_per_sample_output.parent.name}__tau_{tau}.zip"
)
print(file_zip_out)

with zipfile.ZipFile(file_zip_out, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
    for (
        image_id,
        params,
    ) in df.iterrows():
        camera_dict = params.to_dict()

        file_name = "camera_" + image_id.split(".jpg")[0] + ".json"
        fbuffer = StringIO()
        json.dump(camera_dict, fbuffer, indent=4)
        zf.writestr(file_name, fbuffer.getvalue())
        fbuffer.close()
