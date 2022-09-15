#%%
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
sns.set_style("whitegrid")
sns.set_palette("deep")
import pandas as pd

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": "Computer Modern Roman",
        "font.sans-serif": "Computer Modern Sans serif",
    }
)

### %%
taus = [np.inf, 0.019]

loss_keys = ["loss_ndc_total"]
axline_style_kwargs = {"alpha": 0.75, "linestyle": "--"}


dataset_splits_results = [
    (
        Path("experiments/sncalib-test/extrem-gt/per_sample_output.json"),
        "SN-Calib-test: $\mathtt{center}$, GT",
    ),
    (
        Path("experiments/sncalib-test/argmin/argmin/extrem-pred/per_sample_output.json"),
        "SN-Calib-test: $\mathtt{argmin}$, Pred",
    ),
    (
        Path("experiments/sncalib-test/argmin/argmin/extrem-gt/per_sample_output.json"),
        "SN-Calib-test: $\mathtt{argmin}$, GT",
    ),
    (
        Path("experiments/wc14-test/extrem-gt/per_sample_output.json"),
        "WC14-test-center: $\mathtt{center}$, GT",
    ),
    (
        Path("experiments/wc14-test/extrem-pred/per_sample_output.json"),
        "WC14-test-center: $\mathtt{center}$, Pred",
    ),
    (
        Path("experiments/sncalib-valid-cam_center-tau/extrem-pred/per_sample_output.json"),
        "SN-Calib-valid-center: $\mathtt{center}$, Pred",
    ),
]

dfs = []
df_plots = []
for f, split in dataset_splits_results:
    if not f.exists():
        raise FileNotFoundError(f)
    df = pd.read_json(f, orient="records", lines=True)
    df["image_id"] = df["image_ids"].apply(lambda l: l.split(".jpg")[0])
    df["dataset"] = split
    df.set_index("image_id", inplace=True)
    dfs.append(df)
    col_subset = ["loss_ndc_total"]
    df_plot = df.sort_values(by=col_subset[0], ascending=True)[col_subset]
    df_plot["dataset fraction"] = list(range(0, len(df_plot.index)))
    df_plot["dataset fraction"] /= len(df_plot.index)
    df_plot = df_plot.melt(
        value_vars=col_subset,
        value_name="segment reprojection loss",
        var_name="agg",
        id_vars=["dataset fraction"],
    )
    df_plot["dataset"] = split
    df_plots.append(df_plot)


df_plot = pd.concat(df_plots)
df_plot.reset_index(inplace=True)
df = pd.concat(dfs)


# fig, ax = plt.subplots(figsize=(8, 2.5)) # aspect ratio 3.3
# fig, ax = plt.subplots(figsize=(7, 2.1875))  #
g = sns.FacetGrid(
    df_plot,
    hue="dataset",
    height=2.5,
    aspect=2.5,
)
g.map(
    sns.lineplot, "dataset fraction", "segment reprojection loss"
)  # .set(xticks=df_summary_wide.tau.unique())
g.set(ylim=(0.005, 0.03), xlim=[0.0, 0.85])

plt.xticks(np.arange(0, 1.05, 0.1))
plt.yticks(np.arange(0.005, 0.031, 0.01))
plt.legend(frameon=True, title=False, loc="upper left", labelspacing=0.1)

for (
    i,
    tau,
) in enumerate(taus):
    if i == 0:
        continue
    g.facet_axis(0, 0).axhline(y=tau, color=sns.color_palette("dark")[5], **axline_style_kwargs)
# plt.show()
fout = "experiments/ndc_loss_splits.pdf"
print(fout)
plt.savefig(fout, bbox_inches="tight", pad_inches=0.0)
# %%
