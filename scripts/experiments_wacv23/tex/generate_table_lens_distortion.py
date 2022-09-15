#%%
from pathlib import Path
import pandas as pd
import numpy as np

pd.set_option("display.precision", 1)
pd.set_option("display.max_rows", 100)
pd.set_option("display.max_column", 100)
pd.set_option("display.max_colwidth", 1000)


dir_results = Path("experiments/lens_distortion")

# read individual results
dfs = []
for file_summary in dir_results.glob("*/*/eval_calibration_summary.json"):
    df = pd.read_json(file_summary, orient="records", lines=True)
    df["run"] = file_summary.parent.name
    df["dataset"] = file_summary.parent.parent.name
    if "zeta" not in df.columns:
        print("zeta not in header, set to None")
        df["zeta"] = None
    dfs.append(df)
df = pd.concat(dfs)
df.replace([np.NaN], np.inf, inplace=True)
df.sort_values(["dataset", "run", "tau"], ascending=False, inplace=True)
df.reset_index(inplace=True, drop=True)
df

#%%
df_tex = df.copy()
df_tex = df_tex[
    [
        "dataset",
        "run",
        "tau",
        "meanAccuracies@5",
        "meanAccuracies@10",
        "meanAccuracies@20",
        "completeness_subset",
    ]
]
df_tex.rename(
    {
        "dataset": "Dataset",
        "meanAccuracies@5": "AC@5",
        "meanAccuracies@10": "AC@10",
        "meanAccuracies@20": "AC@20",
        "completeness_subset": "CR Subset",
    },
    inplace=True,
    axis=1,
)
df_tex["tau"] = df_tex["tau"].astype(str)
df_tex = df_tex.sort_values(by=["Dataset", "tau", "run"], ascending=[True, False, True])
df_tex

#%%
df_tex["Segmentation"] = df_tex["run"].apply(lambda s: "Pred" if "extrem-pred" in s else "GT")
df_tex["LD"] = df_tex["run"].apply(lambda s: r"\checkmark" if "-ld" in s else r"\xmark")
df_tex = df_tex[
    [
        "Dataset",
        "tau",
        "Segmentation",
        "LD",
        "AC@5",
        "AC@10",
        "AC@20",
        "CR Subset",
    ]
]
df_tex["Dataset"] = df_tex["Dataset"].str.replace("sncalib-valid-center", "SN-Calib-valid-center")
df_tex["Dataset"] = df_tex["Dataset"].str.replace("wc14-test", "WC14-test")
df_tex["tau"] = df_tex["tau"].str.replace("inf", r"$\\infty$")
df_tex.set_index(["Dataset", "tau", "Segmentation"])

#%%
print(
    df_tex.set_index(["Dataset", "Segmentation", "tau"]).to_latex(
        None, index=True, escape=False, header=True, multirow=True
    )
)

# %%
