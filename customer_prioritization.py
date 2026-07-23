import numpy as np
import pandas as pd

deposit_db = pd.read_csv('term-deposit-marketing-2020.csv')

bin_cols = deposit_db.columns[deposit_db.nunique() == 2]
deposit_db[bin_cols] = deposit_db[bin_cols].replace({'yes': 1, 'no': 0})

deposit_db["age_group"] = pd.cut(
    deposit_db["age"],
    bins = [0, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90, np.inf],
    labels = ["<20", "21 - 25", "26 - 30", "31 - 35", "36 - 40",
              "41 - 45", "46 - 50", "51 - 55", "56 - 60", "61 - 70",
              "71 - 80", "81 - 90", ">90"]
)

deposit_db["day_group"] = pd.cut(
    deposit_db["day"],
    bins=[0, 7, 14, 21, 28, np.inf],
    labels=["1 - 7", "8 - 14", "15 - 21", "22 - 28", "29 - 31"]
)

cols = deposit_db.columns

def strata_analysis(data, feature, target):
    overall_prop = data[target].mean()

    results = (data
               .groupby(feature)[target]
               .agg(
                   total_customers = "size",
                   buyers = "sum",
                   purchase_rate = "mean"
               )
               .reset_index()
               )

    results = results[results["total_customers"] > 400]

    results["lift"] = results["purchase_rate"]/overall_prop

    return results.sort_values(
        ["lift", "total_customers"],
        ascending = [False, False]
    )

for col in cols:
    job_segments = strata_analysis(
        deposit_db,
        feature = col,
        target = "y"
    )
    print(job_segments)


segment_results = (
    deposit_db
    .groupby(["job", "marital"])["y"]
    .agg(
        customers="size",
        buyers="sum",
        purchase_rate="mean"
    )
    .reset_index()
)

overall_rate = deposit_db["y"].mean()

segment_results["lift"] = (
    segment_results["purchase_rate"] / overall_rate
)

segment_results = segment_results[
    segment_results["customers"] >= 200
].sort_values("lift", ascending=False)

print(segment_results.head(15))