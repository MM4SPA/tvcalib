#%%
from argparse import ArgumentParser
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option("display.max_rows", 100)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)
pd.set_option("display.colheader_justify", "center")
pd.set_option("display.precision", 3)
sns.set(rc={"xtick.bottom": True, "ytick.left": True})
eval_thresholds = [5, 10, 20]  # in px

args = ArgumentParser()
args.add_argument("--dir_results", type=Path)
args.add_argument("--resolution_width", type=int, default=960)
args.add_argument("--resolution_height", type=int, default=540)
args.add_argument("--from_json", action="store_true")
args.add_argument(
    "--taus",
    nargs="+",
    help="e.g. --taus inf 0.05 0.04 0.03 0.025 0.02 0.017",
    default=["inf", "0.05", "0.04", "0.03", "0.025", "0.02", "0.017", "0.015"],
)
args = args.parse_args()
dir_results = args.dir_results
taus = [float(x) for x in args.taus]
resolution_width = args.resolution_width
resolution_height = args.resolution_height

if not args.from_json:
    files_per_batch = list(sorted([f for f in dir_results.glob("batch_*.pt")]))
    assert len(files_per_batch) > 0


def print_shapes_if_available(x_dict, iter=0):

    tabs = "\t" * iter
    for k, v in x_dict.items():
        if isinstance(v, torch.Tensor):
            print(tabs, k, v.shape)
        elif isinstance(v, dict):
            print(tabs, k)
            print_shapes_if_available(v, iter + 1)
        elif isinstance(v, Iterable):  # all other iterables
            print(tabs, k, v)  # len(v)
        # else:
        #     print(tabs, k, v)


def tensor2list(xdict):
    for k in xdict.keys():
        if isinstance(xdict[k], torch.Tensor):
            xdict[k] = xdict[k].numpy().tolist()
        elif isinstance(xdict[k], dict):
            xdict[k] = tensor2list(xdict[k])
    return xdict


if not args.from_json:
    dataset_dict_stacked = {}
    dataset_dict_stacked["batch_idx"] = []
    for bidx, f in enumerate(files_per_batch):
        per_batch_dict = torch.load(f, map_location=torch.device("cpu"))

        if "per_step_lr" in per_batch_dict:
            del per_batch_dict["per_step_lr"]

        # max distance over all given points -> no need to mask as padded values are 0.0
        per_batch_dict["loss_ndc_lines_distances_max"] = (
            per_batch_dict["loss_ndc_lines_distances_raw"].amax(dim=[-2, -1]).squeeze(-1)
        )
        per_batch_dict["loss_ndc_circles_distances_max"] = (
            per_batch_dict["loss_ndc_circles_distances_raw"].amax(dim=[-2, -1]).squeeze(-1)
        )
        per_batch_dict["loss_ndc_total_max"] = torch.stack(
            [
                per_batch_dict["loss_ndc_lines_distances_max"],
                per_batch_dict["loss_ndc_circles_distances_max"],
            ],
            dim=-1,
        ).max(dim=-1)[0]

        # print_shapes_if_available(per_batch_dict)

        per_batch_dict = tensor2list(per_batch_dict)
        batch_size = len(per_batch_dict["image_ids"])

        per_batch_dict["batch_idx"] = [[str(f)]] * batch_size

        if "time_s" in per_batch_dict:
            per_batch_dict["time_s"] /= batch_size

        if "camera" in per_batch_dict["meta"]:
            del per_batch_dict["meta"]["camera"]
        per_batch_dict.update(per_batch_dict["meta"])
        per_batch_dict.update(per_batch_dict["camera"])
        del per_batch_dict["meta"]
        del per_batch_dict["camera"]

        for k in per_batch_dict.keys():
            if k not in dataset_dict_stacked:
                dataset_dict_stacked[k] = per_batch_dict[k]
            elif isinstance(dataset_dict_stacked[k], list):
                dataset_dict_stacked[k].extend(per_batch_dict[k])
            else:
                dataset_dict_stacked[k] = per_batch_dict[k]

        # for k, v in per_batch_dict.items():
        #     print(k, len(v) if isinstance(v, Iterable) else v)

    df = pd.DataFrame.from_dict(dataset_dict_stacked)
    # explode over t
    explode_cols = [k for k, v in dataset_dict_stacked.items() if isinstance(v, list)]
    df = df.explode(column=explode_cols)
    df["image_id"] = df["image_ids"].apply(lambda l: l.split(".jpg")[0])
    df.set_index("image_id", inplace=True)

    if "match" in df.columns:
        df["stadium"] = df["match"].apply(lambda s: s.split(" - ")[0].strip())
        number_of_images_per_stadium = df.groupby("stadium")["stadium"].agg(len).to_dict()
        df["stadium (number of images)"] = df["stadium"].apply(
            lambda stadium: f"{stadium} ({number_of_images_per_stadium[stadium]})"
        )
    fout = dir_results / "per_sample_output.json"
    df.to_json(fout, orient="records", lines=True)
else:
    fout = dir_results / "per_sample_output.json"
    df = pd.read_json(fout, orient="records", lines=True)
    df["image_id"] = df["image_ids"].apply(lambda l: l.split(".jpg")[0])
    df.set_index("image_id", inplace=True)

print(fout)
print(df)
# TODO:
#%%
# for 980px image width = tau * 480 as NDC_w=2
tau_labels = [f"~{x * (resolution_width / 2):.0f} px" for x in taus]
print(tau_labels)
tau_palette = ["gray", "k", "r", "g", "b", "purple", "yellow", "orange"]
loss_keys = ["loss_ndc_total"]  # ", loss_ndc_total_max"]
axline_style_kwargs = {"alpha": 0.5, "linestyle": "--"}

#%% box plot
if "match" in df.columns:
    for loss_key in loss_keys:
        fig, ax = plt.subplots(figsize=(20, 14))
        # ax.set_xscale("log")
        sns.boxplot(x=loss_key, y="stadium (number of images)", data=df)
        for i, (tau, color, label) in enumerate(zip(taus, tau_palette, tau_labels)):
            if i == 0:
                continue
            ax.axvline(x=tau, label=label, color=color, **axline_style_kwargs)
        plt.legend(title=loss_key)
        plt.xlim([0.0, 0.15])
        fout = dir_results / f"box_per_stadium__{loss_key}.pdf"
        plt.savefig(fout, bbox_inches="tight")
        print(fout)

#%% sorted losses
col_subset = ["loss_ndc_total", "loss_ndc_total_max"]
rename_col_subset = ["mean", "max"]
df_plot = df.sort_values(by=col_subset[0], ascending=True)[col_subset]
df_plot = df_plot.rename(columns={k: v for k, v in zip(col_subset, rename_col_subset)})
df_plot["dataset fraction"] = list(range(0, len(df_plot.index)))
df_plot["dataset fraction"] /= len(df_plot.index)
df_plot = df_plot.melt(
    value_vars=rename_col_subset,
    value_name="reprojection error [NDC]",
    var_name="per-point aggregation",
    id_vars=["dataset fraction"],
)
fig, ax = plt.subplots(figsize=(12, 5))
sns.lineplot(
    x="dataset fraction",
    y="reprojection error [NDC]",
    hue="per-point aggregation",
    data=df_plot,
    ax=ax,
)
plt.xticks(np.arange(0, 1.05, 0.05))
for i, (tau, color) in enumerate(zip(taus, tau_palette)):
    if i == 0:
        continue
    ax.axhline(y=tau, color=color, **axline_style_kwargs)

plt.ylim([0.0, 0.1])
plt.yticks(np.arange(0, 0.105, 0.01))
fout = dir_results / f"reconstruction_error_sorted_{col_subset[0]}.pdf"
plt.savefig(fout, bbox_inches="tight")
print(fout)
