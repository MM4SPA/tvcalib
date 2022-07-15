from SoccerNet.Evaluation.FieldCalibration import evaluate
from pprint import pprint
import zipfile
from io import StringIO
import tempfile
import json
from pathlib import Path
from tqdm.auto import tqdm

split = "test"

gt_annotation_dir = Path(f"/nfs/data/soccernet/calibration/{split}")
gt_annotations = [f for f in gt_annotation_dir.glob("*.json")]
gt_annotations = [f for f in gt_annotations if "match_info" not in f.name]
assert len(gt_annotations) > 0
# assert len(gt_annotations) == 2719  # test split


# file_zip_out = tempfile.NamedTemporaryFile(mode="w")
file_zip_out = gt_annotation_dir.parent / f"{split}_json_annotations.zip"
print(file_zip_out)

with zipfile.ZipFile(file_zip_out, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:

    for f in tqdm(gt_annotations):
        fbuffer = StringIO()
        with open(f) as fr:
            d = json.load(fr)
        json.dump(d, fbuffer, indent=4)
        zf.writestr(f"{split}/" + f.name, fbuffer.getvalue())

# pred_dir = Path(f"/nfs/home/rhotertj/datasets/sn-calib-challenge_endpoints_4_30/{split}")
pred_dir = Path(f"/nfs/home/rhotertj/datasets/sn-calib-challenge_4_30/{split}")

predicted_files = [f for f in pred_dir.glob("*.json")]

file_zip_predicted = "tmp.zip"
with zipfile.ZipFile(file_zip_predicted, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
    for f in tqdm(predicted_files):
        fbuffer = StringIO()
        with open(f) as fr:
            d = json.load(fr)
        json.dump(d, fbuffer, indent=4)
        zf.writestr(f.name, fbuffer.getvalue())


mean_accuracies = []
for t in [5, 10, 20]:
    print(t)
    result = evaluate(file_zip_out, file_zip_predicted, threshold=t)
    pprint(result)
    mean_accuracies.append((t, result["meanAccuracies"]))
print(mean_accuracies)
