#%%
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

sns.set()
pd.set_option("display.precision", 1)
pd.set_option("display.max_rows", 100)
pd.set_option("display.max_column", 100)
pd.set_option("display.max_colwidth", 1000)

POS_INF_REPLACE = 0.0
POS_INF_REMOVE = False

dir_results = Path("experiments/sncalib-test-cam_center")

# read individual results
dfs = []
for file_summary in dir_results.glob("*/eval_calibration_summary.json"):
    print(file_summary)
    df = pd.read_json(file_summary, orient="records", lines=True)
    print(file_summary.parent.name)
    df["run"] = file_summary.parent.name
    if "chen" in file_summary.parent.name:
        df["tau"] = 0.0
    if "zeta" not in df.columns:
        print("zeta not in header, set to None")
        # print(df.columns)
        df["zeta"] = None
    dfs.append(df)

df = pd.concat(dfs)
df.sort_values(["run", "tau"], ascending=False, inplace=True)
df.reset_index(inplace=True, drop=True)
df.replace([np.NaN], np.inf, inplace=True)
if POS_INF_REMOVE:
    df = df.loc[~(df.tau == np.inf)]
else:
    df = df.replace([np.inf, float("inf")], POS_INF_REPLACE)

markers = ["D" if "extrem-gt" in s else "X" for s in df["run"].unique()]

df.rename(
    {
        "meanAccuracies@5": "AC@5",
        "meanAccuracies@10": "AC@10",
        "meanAccuracies@20": "AC@20",
        "compound_score": "CS",
        "compound_score_subset": "CS_subset",
        "completeness": "Completeness",
        "completeness_subset": "Completeness Subset",
    },
    inplace=True,
    axis=1,
)
df["Segmentation"] = df["run"].apply(lambda s: s.split("extrem")[-1][1:])
df["Segmentation"] = df["Segmentation"].apply(lambda s: s.title() if s != "gt" else "GT")
df.sort_values(
    ["project_from", "evaluate_planar", "Segmentation", "tau", "zeta"], ascending=True, inplace=True
)

df = df[
    [
        "run",
        "project_from",
        "evaluate_planar",
        "tau",
        "zeta",
        "Segmentation",
        "AC@5",
        "AC@10",
        "AC@20",
        "Completeness Subset",
        "CS_subset",
    ]
]
print(df)
#%%
df_tex = df.copy()
df_tex["run"] = (
    df_tex["project_from"]
    + "__"
    + df_tex["run"]
    + "__tau"
    + df_tex["tau"].astype(str)
    + "__zeta"
    + df_tex["zeta"].astype(str)
)


table_splits = {
    "calibration and gt": ((df_tex.evaluate_planar == False) & (df_tex.Segmentation == "GT")),
    "calibration and pred": ((df_tex.evaluate_planar == False) & (df_tex.Segmentation != "GT")),
    "homography and gt": ((df_tex.evaluate_planar == True) & (df_tex.Segmentation == "GT")),
    "homography and pred": ((df_tex.evaluate_planar == True) & (df_tex.Segmentation != "GT")),
}
pprint(dict(defaultdict(list, {k: None for k in df_tex["run"].unique().tolist()})))

#%%
run_rename = {
    "Camera__extrem-gt__tau0.0__zeta0.0": r"\acrshort{method}",
    "Camera__extrem-gt__tau0.019__zeta0.0": r"\acrshort{method}($\tau$)",
    "Camera__extrem-pred__tau0.0__zeta0.0": r"\acrshort{method}",
    "Camera__extrem-pred__tau0.019__zeta0.0": r"\acrshort{method}($\tau$)",
    "HDecomp__chen-n_fl-50k-extrem-gt__tau0.0__zeta100.0": r"HDecomp + Chen \& Little~\cite{chen2019sports}",
    "HDecomp__chen-n_fl-50k-extrem-pred__tau0.0__zeta100.0": r"HDecomp + Chen \& Little~\cite{chen2019sports}",
    "HDecomp__chen-u_fl-u_xyz-50k-extrem-gt__tau0.0__zeta100.0": r"HDecomp + \cite{chen2019sports} ($\mathcal{U}_{FoV}$+$\mathcal{U}_{xyz}$)",
    "HDecomp__chen-u_fl-u_xyz-50k-extrem-pred__tau0.0__zeta100.0": r"HDecomp + \cite{chen2019sports} ($\mathcal{U}_{FoV}$+$\mathcal{U}_{xyz}$)",
    "HDecomp__Hline-extrem-gt__tau0.0__zeta100.0": r"HDecomp + DLT Lines",
    "HDecomp__Hline-extrem-pred__tau0.0__zeta100.0": r"HDecomp + DLT Lines",
    "Homography__chen-n_fl-50k-extrem-gt__tau0.0__zeta0.0": r"Chen \& Little~\cite{chen2019sports}",
    "Homography__chen-n_fl-50k-extrem-pred__tau0.0__zeta0.0": r"Chen \& Little~\cite{chen2019sports}",
    "Homography__chen-u_fl-u_xyz-50k-extrem-gt__tau0.0__zeta0.0": r"\cite{chen2019sports} ($\mathcal{U}_{FoV}$+$\mathcal{U}_{xyz}$)",
    "Homography__chen-u_fl-u_xyz-50k-extrem-pred__tau0.0__zeta0.0": r"\cite{chen2019sports} ($\mathcal{U}_{FoV}$+$\mathcal{U}_{xyz}$)",
    "Homography__Hline-extrem-gt__tau0.0__zeta0.0": r"DLT Lines",
    "Homography__Hline-extrem-pred__tau0.0__zeta0.0": r"DLT Lines",
}
df_tex["run"] = df_tex["run"].apply(lambda run: run_rename[run])
df_tex["run"] = df_tex["run"].apply(
    lambda run: r"\rowcolor{OliveGreen!10} " + run if r"{method}" in run else run
)
#%%


for k, table_split in table_splits.items():
    df_tex_subset = df_tex.loc[table_split].sort_values(by=["CS_subset"], ascending=[False])[
        ["run", "Segmentation", "AC@5", "AC@10", "AC@20", "Completeness Subset", "CS_subset"]
    ]
    print("% %%%%%%%%%%%%%%%%%%%%", k, "%%%%%%%%%%%%%%%%%%%%%%%%")
    # print(df_tex_subset.style.format(precision=1).hide(axis='index').to_latex(None))
    print(df_tex_subset.to_latex(None, index=False, escape=False, header=False))

# %%
