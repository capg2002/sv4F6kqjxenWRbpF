import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    train_test_split,
    cross_validate,
    GridSearchCV,
)

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss
)

from sklearn.feature_selection import RFECV

from sklearn.pipeline import Pipeline

# Note 15000 iterations was tested, and the results were identical to 
# having 5000 iterations.
seed = 22
iterations = 2000

deposit_db = pd.read_csv('term-deposit-marketing-2020.csv')

bin_cols = deposit_db.columns[deposit_db.nunique() == 2]
deposit_db[bin_cols] = deposit_db[bin_cols].replace({'yes': 1, 'no': 0})

factor_cols = deposit_db.select_dtypes(include = ['object', 'category'])
factor_cols = factor_cols.columns[factor_cols.nunique() > 2]

print(factor_cols)
deposit_db = pd.get_dummies(
    deposit_db,
    columns=factor_cols,
    drop_first = True,
    dtype=int)



Y_var = deposit_db["y"]
X_full = deposit_db.drop("y", axis=1)

cv = StratifiedKFold(n_splits=5,
    shuffle=True,
    random_state=seed
    )
    
X_train_f, X_test_f, Y_train, Y_test = train_test_split(X_full, 
            Y_var, test_size=0.2, random_state=seed)

full_model = LogisticRegression(max_iter = iterations)

full_model.fit(X_train_f, Y_train)
Y_pred_f = full_model.predict(X_test_f)
Y_prbs_f = full_model.predict_proba(X_test_f)[:,1]

scores_full = cross_validate(
    full_model,
    X_full,
    Y_var,
    cv=cv,
    scoring=["balanced_accuracy", "accuracy", "precision", "recall", "f1", "roc_auc", "neg_log_loss"]
)

scores_full_names = list(scores_full.keys())

for name in scores_full_names:
    print("Full", name, ":", scores_full[name].mean())

reduced_model = RFECV(
    estimator=full_model,
    step=1,
    cv=cv,
    scoring="balanced_accuracy",
    min_features_to_select=1
)

reduced_model.fit(X_train_f, Y_train)

selected_features = X_full.columns[reduced_model.support_]
X_reduced = X_full[selected_features]

print("Selected features:")
print(selected_features)

final_reduced_model = LogisticRegression(
    max_iter=iterations
)

scores_reduced = cross_validate(
    final_reduced_model,
    X_reduced,
    Y_var,
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

for name, values in scores_reduced.items():
    print("Reduced", name, ":", values.mean())

print("Number of observations:", X_full.shape[0])
print("Number of original features:", X_full.shape[1])
print("Number of selected features:", X_reduced.shape[1])

## BALANCED

full_model = LogisticRegression(class_weight="balanced",max_iter = iterations)

full_model.fit(X_train_f, Y_train)
Y_pred_f = full_model.predict(X_test_f)
Y_prbs_f = full_model.predict_proba(X_test_f)[:,1]

scores_full = cross_validate(
    full_model,
    X_full,
    Y_var,
    cv=cv,
    scoring=["balanced_accuracy", "accuracy", "precision", "recall", "f1", "roc_auc", "neg_log_loss"]
)

scores_full_names = list(scores_full.keys())

for name in scores_full_names:
    print("Full", name, ":", scores_full[name].mean())

reduced_model = RFECV(
    estimator=full_model,
    step=1,
    cv=cv,
    scoring="balanced_accuracy",
    min_features_to_select=1
)

reduced_model.fit(X_train_f, Y_train)

selected_features = X_full.columns[reduced_model.support_]
X_reduced = X_full[selected_features]

print("Selected features:")
print(selected_features)

final_reduced_model = LogisticRegression(
    class_weight="balanced",
    max_iter=iterations
)

scores_reduced = cross_validate(
    final_reduced_model,
    X_reduced,
    Y_var,
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

for name, values in scores_reduced.items():
    print("Reduced", name, ":", values.mean())

print("Number of observations:", X_full.shape[0])
print("Number of original features:", X_full.shape[1])
print("Number of selected features:", X_reduced.shape[1])

# It is noted that the full balanced model already has high balanced accuracy
# at 86.46% with high recall 86.12%, which displays a preferred model over
# the reduced unbalanced model, which had 63.42% balanced accuracy with
# extremely low recall at 28.24%, showing how the unbalanced model is
# too conservative at claiming purchases.

# RFECV was used for both, with the balanced model not being reduced at all.
# Separate analyses must be done to determine a minimal feature set, and 
# finding segments to prioritize.
