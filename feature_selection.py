# Illustrating which features are most impactful using L1 loss.

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss
)

import pandas as pd
import numpy as np
import textwrap
import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold,
    train_test_split,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Note 15000 iterations was tested, and the results were identical to 
# having 5000 iterations.
seed = 22
iterations = 2000

# Set up for dataframe
deposit_db = pd.read_csv('term-deposit-marketing-2020.csv')

bin_cols = deposit_db.columns[deposit_db.nunique() == 2]
deposit_db[bin_cols] = deposit_db[bin_cols].replace({'yes': 1, 'no': 0})

factor_cols = deposit_db.select_dtypes(include = ['object', 'category'])
factor_cols = factor_cols.columns[factor_cols.nunique() > 2]

deposit_db = pd.get_dummies(
    deposit_db,
    columns=factor_cols,
    drop_first = True,
    dtype=int)

# Set up for fitting
Y_var = deposit_db["y"]
X_full = deposit_db.drop("y", axis=1)

cv = StratifiedKFold(n_splits=5,
    shuffle=True,
    random_state=seed
    )
    
X_train_f, X_test_f, Y_train, Y_test = train_test_split(X_full, 
            Y_var, test_size=0.2, random_state=seed)

# Eligible C values for L1 loss model selection.
C_values = np.logspace(-6, 0, 30)

results = []

# Stores the features selected for the previous C value.
previous_selected_features = set()

for position, C in enumerate(C_values):

    # Scale and fit logistic regression
    model = Pipeline([
        ("scaler", StandardScaler()),
        (
            "logistic",
            LogisticRegression(
                penalty="l1",
                solver="liblinear",
                C=C,
                class_weight="balanced",
                max_iter=5000
            )
        )
    ])

    # Cross validate model
    scores = cross_validate(
        model,
        X_train_f,
        Y_train,
        cv=cv,
        scoring=["neg_log_loss", "balanced_accuracy"],
        n_jobs=-1
    )
    model.fit(X_train_f, Y_train)

    # Confirm coefficients
    coefficients = model.named_steps["logistic"].coef_[0]

    # Remove coefficients shrunk close to 0 by L1 fitting.
    selected_mask = np.abs(coefficients) > 1e-8

    selected_features = X_train_f.columns[
        selected_mask
    ].tolist()

    current_selected_features = set(selected_features)

    # Compare the current C model with the previous C model. This
    # is in order to tell which features were added and removed in each step.

    if position == 0:
        added_features = current_selected_features
        removed_features = set()
    else:
        added_features = (
            current_selected_features
            - previous_selected_features
        )

        removed_features = (
            previous_selected_features
            - current_selected_features
        )

    results.append({
        "C": C,
        "features_selected": len(selected_features),

        "selected_feature_names": ", ".join(
            sorted(current_selected_features)
        ),

        "features_added_from_previous_C": ", ".join(
            sorted(added_features)
        ),

        "features_removed_from_previous_C": ", ".join(
            sorted(removed_features)
        ),

        "mean_neg_log_loss": scores["test_neg_log_loss"].mean(),
        "sd_neg_log_loss": scores["test_neg_log_loss"].std(),
        "mean_balanced_accuracy": scores["test_balanced_accuracy"].mean(),
        "sd_balanced_accuracy": scores["test_balanced_accuracy"].std()
    })

    # Save the current set for the next iteration.
    previous_selected_features = current_selected_features.copy()


selection_results = pd.DataFrame(results)

selection_results = selection_results.sort_values(
    ["features_selected", "mean_balanced_accuracy"],
    ascending=[True, False]
)

# Convert negative log loss into ordinary positive log loss.
selection_results["log_loss"] = (
    -selection_results["mean_neg_log_loss"]
)

# Improvement relative to the previous, smaller C.
selection_results["improvement_from_previous"] = (
    selection_results["log_loss"].shift(1)
    - selection_results["log_loss"]
)

# Count added and removed features.
selection_results["features_added_count"] = (
    selection_results["features_added_from_previous_C"]
    .apply(
        lambda x: 0
        if not x
        else len(x.split(", "))
    )
)

selection_results["features_removed_count"] = (
    selection_results["features_removed_from_previous_C"]
    .apply(
        lambda x: 0
        if not x
        else len(x.split(", "))
    )
)

# Compact table for comparing models
compact_table = selection_results[
    [
        "C",
        "features_selected",
        "log_loss",
        "mean_balanced_accuracy",
        "improvement_from_previous",
        "features_added_count",
        "features_removed_count"
    ]
].copy()

compact_table.columns = [
    "C",
    "Features",
    "Log loss",
    "Balanced accuracy",
    "Improvement",
    "Added",
    "Removed"
]


# All code below is my attempt to make a more legible table for comparisons.
print("\nMODEL COMPARISON")
print("-" * 82)

print(
    compact_table.to_string(
        index=False,
        formatters={
            "C": lambda x: f"{x:.6f}",
            "Log loss": lambda x: f"{x:.6f}",
            "Improvement": lambda x: (
                "—" if pd.isna(x) else f"{x:+.6f}"
            )
        }
    )
)

print("\n\nFEATURE SELECTION DETAILS")
print("=" * 82)

for _, row in selection_results.iterrows():

    selected = (
        row["selected_feature_names"]
        if row["selected_feature_names"]
        else "None"
    )

    added = (
        row["features_added_from_previous_C"]
        if row["features_added_from_previous_C"]
        else "None"
    )

    removed = (
        row["features_removed_from_previous_C"]
        if row["features_removed_from_previous_C"]
        else "None"
    )

    print(f"\nC = {row['C']:.6f}")
    print(f"Features selected: {row['features_selected']}")
    print(f"Log loss: {-row['mean_neg_log_loss']:.6f}")

    print("\nAdded since previous C:")
    print(
        textwrap.fill(
            added,
            width=78,
            initial_indent="  ",
            subsequent_indent="  "
        )
    )

    print("\nRemoved since previous C:")
    print(
        textwrap.fill(
            removed,
            width=78,
            initial_indent="  ",
            subsequent_indent="  "
        )
    )

    print("\nAll selected features:")
    print(
        textwrap.fill(
            selected,
            width=78,
            initial_indent="  ",
            subsequent_indent="  "
        )
    )

    print("-" * 82)

# Chosen C which sacrifices variables while still yielding a high
# balanced accuracy. 

chosen_C = 0.00329

chosen_model = Pipeline([
    ("scaler", StandardScaler()),
    ("logistic", LogisticRegression(
        penalty="l1",
        solver="liblinear",
        C=chosen_C,
        class_weight="balanced",
        max_iter=5000
    ))
])

# Diagnostics cross validated
chosen_scores = cross_validate(
    chosen_model,
    X_train_f,
    Y_train,
    cv=cv,
    scoring=[
        "balanced_accuracy",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "neg_log_loss"
    ]
)

# Print mean diagnostics
for name, values in chosen_scores.items():
    print(name, ":", values.mean())

chosen_model.fit(X_train_f, Y_train)

coefficients = chosen_model.named_steps["logistic"].coef_[0]

# Remove significantly small coefficients.
selected_features = X_train_f.columns[
    np.abs(coefficients) > 1e-8
]

print("Selected features:")
print(selected_features.tolist())


Y_pred_f = chosen_model.predict(X_test_f)
Y_prbs_f = chosen_model.predict_proba(X_test_f)[:,1]

print("FINAL TEST RESULTS")
print("Accuracy:", accuracy_score(Y_test, Y_pred_f))
print("Balanced accuracy:", balanced_accuracy_score(Y_test, Y_pred_f))
print("Precision:", precision_score(Y_test, Y_pred_f))
print("Recall:", recall_score(Y_test, Y_pred_f))
print("F1:", f1_score(Y_test, Y_pred_f))
print("ROC AUC:", roc_auc_score(Y_test, Y_prbs_f))
print("Log loss:", log_loss(Y_test, Y_prbs_f))


# Variables were introduced in the following order:

# duration,

# month,

# contact,

# housing,

# job, marital,

# education, loan,

# balance, campaign, day,

# default,

# age

# Duration contains the strongest initial predictive signal.
# Month and contact method provide the next-largest useful additions.
# Housing, job, and marital status add further moderate information.
# Variables entering later provide progressively smaller incremental improvements.

# For the selected reduced model, it has 21 features instead of 36.
# It includes the following variables:
# Balance, Housing, Loan, Day, Duration, Campaign, "has blue collar", "has retired",
# "has services", "has student", "is married", "has tertiary edu", "does not have unknown contact",
# "August", "Feb", "Jan", "Jul", "Mar", "May", "Nov", "Oct"

# The model has 83.94% balanced accuracy compared to the 84.31% seen in the initial reg model,
# which is marginal considering 15 variables are dropped.