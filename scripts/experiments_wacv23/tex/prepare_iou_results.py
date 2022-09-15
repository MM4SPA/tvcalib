
#%%
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import seaborn as sns

from evaluation.iou.iou_eval import iou_whole, iou_part

sns.set()
image_height = 720
image_width= 1280

file_per_sample_gt = Path("experiments/wc14-test/h_manual-extrem_gt/per_sample_output.json")
assert file_per_sample_gt.exists()

per_sample_outputs = [
    f for f in Path("experiments/wc14-test").glob("*/**/per_sample_output.json")
]


dfs_stacked = []
for file_per_sample_predictions in per_sample_outputs:
    print(file_per_sample_predictions.absolute())
    assert file_per_sample_predictions.exists()


    df_gt = pd.read_json(file_per_sample_gt, orient="records", lines=True)
    df_gt["homography"] = df_gt["homography"].apply(lambda h: np.array(h))
    df_gt.set_index("image_ids", inplace=True)

    df_pred = pd.read_json(file_per_sample_predictions, orient="records", lines=True)
    df_pred["homography"] = df_pred["homography"].apply(lambda h: np.array(h))
    df_pred.set_index("image_ids", inplace=True)

    ious_whole = []
    ious_part = []
    image_ids_eval = []
    for i, image_id in enumerate(df_gt.index):
        
        if image_id not in df_pred.index:
            print("skip", image_id)
            # calculate only for a subset if pred not available!
            continue

        H_pred_sn = torch.from_numpy(df_pred.loc[image_id]["homography"]).unsqueeze(0).float()
        H_gt_sn = torch.from_numpy(df_gt.loc[image_id]["homography"]).unsqueeze(0).float()

        iou_whole_tensor, img_composite_whole = iou_whole(H_pred_sn, H_gt_sn, scaling_factor=2)
        iou_part_tensor, img_composite_part = iou_part(H_pred_sn, H_gt_sn, image_height, image_width)
        ious_whole.append(iou_whole_tensor.squeeze().item())
        ious_part.append(iou_part_tensor.squeeze().item())
        image_ids_eval.append(image_id)

    
    df_iou = pd.DataFrame(data={"image_ids": image_ids_eval, "iou_whole": ious_whole, "iou_part": ious_part})
    df_iou = df_iou.sort_values(by="iou_part")
    df_iou["image_id_index"] = list(range(len(df_iou.index)))

    run = file_per_sample_predictions.parent.name
    print("whole", "mean", f'{df_iou["iou_whole"].mean()*100:.2f}', "median", f'{df_iou["iou_whole"].median()*100:.2f}')
    print("part", "mean", f'{df_iou["iou_part"].mean()*100:.2f}', "median", f'{df_iou["iou_part"].median()*100:.2f}')

    df_iou = df_iou.melt(["image_id_index", "image_ids"])
    df_iou["run"] = run
    dfs_stacked.append(df_iou)

dfs_stacked = pd.concat(dfs_stacked)
dfs_stacked = dfs_stacked.reset_index(drop=True)
dfs_stacked
# %%
g = sns.FacetGrid(dfs_stacked, col="run", hue="variable") # height=2.5, sharey=False, legend_out=True, aspect=1.8,
g.map_dataframe(sns.scatterplot, x="image_id_index", y="value")
# %%
dfs_stacked.groupby(["run", "variable"]).agg({"value": [np.mean, np.median, np.std]})
print(dfs_stacked)
# %%
dfs_stacked.loc[dfs_stacked.variable == "iou_part"].groupby("run").agg({"value": [np.mean, np.median]}) * 100
# %%
