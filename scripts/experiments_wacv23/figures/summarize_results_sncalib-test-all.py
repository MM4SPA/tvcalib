
#%%
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns; sns.set()
sns.set_style("whitegrid")
sns.set_palette("Paired")
pd.set_option("display.precision", 3)
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": "Computer Modern Roman",
        "font.sans-serif": "Computer Modern Sans serif",
    }
)
POS_INF_REMOVE = True
POS_INF_REPLACE = 0.0 # irrelevant if POS_INF_REMOVE=True

#%%
dir_results_candidates = [
    Path("experiments/sncalib-test") / "argmin/argmin/extrem-gt",
    Path("experiments/sncalib-test") / "stacked/stacked_extrem-gt_cam-gt", 
    Path("experiments/sncalib-test") / "extrem-gt", # main tribune as initialization
    Path("experiments/sncalib-test") / "argmin/argmin/extrem-pred",
    Path("experiments/sncalib-test") / "stacked/stacked_extrem-pred_cam-gt", 
    Path("experiments/sncalib-test") / "center_extrem-gt_cam-gt",  # center as init + GT camera center filtering
    # Path("experiments/sncalib-test-cam_center") / "chen-n_fl-50k-extrem-gt",
    # Path("experiments/sncalib-test-cam_center") / "chen-n_fl-50k-extrem-pred",
    Path("experiments/sncalib-test-cam_center") / "chen-u_fl-u_xyz-50k-extrem-gt", # used as best variant
    # Path("experiments/sncalib-test-cam_center") / "chen-u_fl-u_xyz-50k-extrem-pred",
]

run_rename = {
    "sncalib-test/extrem-gt": "\emph{TVCalib}($\mathtt{center}$) GT",
    "sncalib-test/center_extrem-gt_cam-gt": "\emph{TVCalib}($\mathtt{center}$) GT (center)",
    'stacked/stacked_extrem-gt_cam-gt': "\emph{TVCalib}($\mathtt{stacked}$) GT", 
    'stacked/stacked_extrem-pred_cam-gt': "\emph{TVCalib}($\mathtt{stacked}$) Pred", 
    'argmin/extrem-gt': "\emph{TVCalib}($\mathtt{argmin}$) GT", 
    'argmin/extrem-pred': "\emph{TVCalib}($\mathtt{argmin}$) Pred", 
    'sncalib-test-cam_center/chen-n_fl-50k-extrem-gt': r'HDecomp + Chen and Little GT',
    'sncalib-test-cam_center/chen-n_fl-50k-extrem-pred': r'HDecomp + Chen and Little Pred',
    'sncalib-test-cam_center/chen-u_fl-u_xyz-50k-extrem-gt': r'Chen and Little ($\mathcal{U}_{FoV}$+$\mathcal{U}_{xyz}$) GT',
    'sncalib-test-cam_center/chen-u_fl-u_xyz-50k-extrem-pred': r'HDecomp + Chen and Little ($\mathcal{U}_{FoV}$+$\mathcal{U}_{xyz}$) Pred',
}

# read individual results
dfs = []
for result_dir in dir_results_candidates:
    file_summary = result_dir / "eval_calibration_summary.json"
    if not file_summary.exists():
        raise FileNotFoundError(file_summary)
    df = pd.read_json(file_summary, orient="records", lines=True)
    df["run"] = result_dir.parent.name + "/" + result_dir.name
    if "chen" in result_dir.name:
        # select only the variant where we evalute the calibration performance
        df = df.loc[(df.project_from == "HDecomp") & (df.evaluate_planar == False)]
        df["tau"] = 0.0
    else:
        df = df.loc[(df.project_from == "Camera") & (df.evaluate_planar == False)]
    dfs.append(df)

df = pd.concat(dfs)
df.sort_values(["run", "tau"], ascending=False, inplace=True)
df.reset_index(inplace=True, drop=True)
df.replace([np.NaN], np.inf, inplace=True)
if POS_INF_REMOVE:
    df = df.loc[~(df.tau == np.inf)]
else:
    df = df.replace([np.inf, float("inf")], POS_INF_REPLACE)

df["run"] = df["run"].apply(lambda run: run_rename[run])
df.rename({"meanAccuracies@5": "AC@5"}, inplace=True, axis=1)
df.rename({"meanAccuracies@10": "AC@10"}, inplace=True, axis=1)
df.rename({"meanAccuracies@20": "AC@20"}, inplace=True, axis=1)
df.rename({"compound_score": "CS"}, inplace=True, axis=1)
df.rename({"compound_score_subset": "CS_subset"}, inplace=True, axis=1)
df.rename({"completeness": "Completeness"}, inplace=True, axis=1)
df.rename({"completeness_subset": "Completeness Subset"}, inplace=True, axis=1)

df.sort_values(["project_from", "run"], ascending=False, inplace=True)
# df.sort_values("CS", ascending=False, inplace=True)
print(df)
df
#%%
markers = ["D" if s.endswith("GT") else "X" for s in df["run"].unique()]
label_order = [
    "\emph{TVCalib}($\mathtt{argmin}$) GT", 
    "\emph{TVCalib}($\mathtt{argmin}$) Pred", 
    "\emph{TVCalib}($\mathtt{stacked}$) GT", 
    "\emph{TVCalib}($\mathtt{stacked}$) Pred",
    "\emph{TVCalib}($\mathtt{center}$) GT",
    "\emph{TVCalib}($\mathtt{center}$) GT (center)",
    # r'HDecomp + Chen and Little GT',
    # r'HDecomp + Chen and Little Pred',
    r'Chen and Little ($\mathcal{U}_{FoV}$+$\mathcal{U}_{xyz}$) GT',
    # r'HDecomp + Chen and Little ($\mathcal{U}_{FoV}$+$\mathcal{U}_{xyz}$) Pred',
]


g = sns.FacetGrid(df, hue="run", height=3.0, sharey=True, aspect=2.5, legend_out=False, hue_kws={"marker": markers}, palette="Paired").set(ylim=[62., 85])
g.map(sns.regplot, "tau", "CS", order=5, ci=None)
g.add_legend(markerscale=2, frameon=True, title="", loc=[-0.01, 0.2], label_order=label_order)
g.set(xlabel=r"$\tau$")

fout = Path("experiments") / "sncalib-test-all-tau_vs_eval_score.pdf"
print(fout)
g.savefig(fout, bbox_inches='tight', pad_inches=0)

#%%
df_acc_long = pd.melt(df, id_vars=["run", "tau", "Completeness", "Completeness Subset"], value_vars=["AC@5", "AC@10", "AC@20"], value_name="AC", var_name="t")
df_acc_long["t"] = df_acc_long["t"].apply(lambda s: s.split("@")[-1] + " px")
df_acc_long["Missed (1 - Completeness) [\%]"] = 100 - df_acc_long["Completeness"]
markers = ["D" if s.endswith("GT") else "X" for s in df_acc_long["run"].unique()]
print(df_acc_long)
df_acc_long
#%%
g = sns.FacetGrid(df_acc_long, col="t", hue="run", height=2.5, sharey=False, legend_out=True, aspect=1.8, hue_kws={"marker": markers})
g.map(sns.regplot, "Missed (1 - Completeness) [\%]", "AC", scatter_kws={"s": 50}, order=2, ci=None)
g.add_legend(markerscale=1.1, frameon=True, title="")
sns.move_legend(g, "center right")

for ax in g.axes.flat:
    print()
    ax.axvline(x=100-71.5, color="k", **{"alpha": 0.5, "linestyle": "--"}) # max expected completeness for main camera (center, left, right)
