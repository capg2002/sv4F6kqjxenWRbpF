# Initial Logistic regression attempt

import pandas as pd
import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold,
    train_test_split,
    cross_validate,
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

warnings.filterwarnings('ignore')

# Note 15000 iterations was tested, and the results were identical to 
# having 5000 iterations.
seed = 22
iterations = 2000

# Loading dataset.
deposit_db = pd.read_csv('term-deposit-marketing-2020.csv')

# Transforming binary columns into numeric.
deposit_db["y"] = deposit_db["y"].replace({'yes': 1, 'no': 0})

# Transforming cat columns into factors.
factor_cols = deposit_db.select_dtypes(include = ['object', 'category'])
factor_cols = factor_cols.columns

print(factor_cols)
deposit_db = pd.get_dummies(
    deposit_db,
    columns=factor_cols,
    drop_first = True,
    dtype=int)

# Setting up for logistic regression models
Y_var = deposit_db["y"]
X_full = deposit_db.drop("y", axis=1)

cv = StratifiedKFold(n_splits=5,
    shuffle=True,
    random_state=seed
    )

# Train-test split.
X_train_f, X_test_f, Y_train, Y_test = train_test_split(X_full, 
            Y_var, test_size=0.2, random_state=seed)

# Full unbalanced model.
full_model = LogisticRegression(max_iter = iterations)
full_model.fit(X_train_f, Y_train)

# Predicted values
Y_pred_f = full_model.predict(X_test_f)
Y_prbs_f = full_model.predict_proba(X_test_f)[:,1]

# CV full model reporting average scoring metrics
scores_full = cross_validate(
    full_model,
    X_train_f,
    Y_train,
    cv=cv,
    scoring=["balanced_accuracy", "accuracy", "precision", "recall", 
             "f1", "roc_auc", "neg_log_loss"]
)

print("FINAL TEST RESULTS")
print("Accuracy:", accuracy_score(Y_test, Y_pred_f))
print("Balanced accuracy:", balanced_accuracy_score(Y_test, Y_pred_f))
print("Precision:", precision_score(Y_test, Y_pred_f))
print("Recall:", recall_score(Y_test, Y_pred_f))
print("F1:", f1_score(Y_test, Y_pred_f))
print("ROC AUC:", roc_auc_score(Y_test, Y_prbs_f))
print("Log loss:", log_loss(Y_test, Y_prbs_f))


# Printing diagnostics
scores_full_names = list(scores_full.keys())
for name in scores_full_names:
    print("Full", name, ":", scores_full[name].mean())

# Accuracy is high, at ~93%, but the balanced accuracy is ~62%.
# Note that 92.76% of the dataset is negatives, so this likely predicts
# the majority of the negatives correctly, but also predicts false negatives often. 

# This is corroborated by the recall at ~25%, (TP/(TP + FN)) showing how a 
# the model misses actual positive cases, resulting in a high number of false negatives.

# This suggests that the model over-relies on predicting negatives.

# RFECV selection did removed two variables. Marginally improving the model.
# Can be ignored. Feature selection done in full in feature_selection.py
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

## BALANCED LOGISTIC REGRESSION (fixes loss function to be weighted)

# Fitting full balanced model.
full_model = LogisticRegression(class_weight="balanced",max_iter = iterations)
full_model.fit(X_train_f, Y_train)
Y_pred_f = full_model.predict(X_test_f)
Y_prbs_f = full_model.predict_proba(X_test_f)[:,1]

# Scores for full balanced model.
scores_full = cross_validate(
    full_model,
    X_train_f,
    Y_train,
    cv=cv,
    scoring=["balanced_accuracy", "accuracy", "precision", "recall", "f1", "roc_auc", "neg_log_loss"]
)

scores_full_names = list(scores_full.keys())
for name in scores_full_names:
    print("Full", name, ":", scores_full[name].mean())

Y_pred_f = full_model.predict(X_test_f)
Y_prbs_f = full_model.predict_proba(X_test_f)[:,1]

print("FINAL TEST RESULTS")
print("Accuracy:", accuracy_score(Y_test, Y_pred_f))
print("Balanced accuracy:", balanced_accuracy_score(Y_test, Y_pred_f))
print("Precision:", precision_score(Y_test, Y_pred_f))
print("Recall:", recall_score(Y_test, Y_pred_f))
print("F1:", f1_score(Y_test, Y_pred_f))
print("ROC AUC:", roc_auc_score(Y_test, Y_prbs_f))
print("Log loss:", log_loss(Y_test, Y_prbs_f))


# Notably, this full balanced model is considerably more consistent, improving
# balanced accuracy and recall significantly. 

# Balanced accuracy is ~84% and recall is ~82%, which is very strong.
# Notably, accuracy and precision have decreasing. It makes sense that 
# precision decreases, as it shows that, when the model says "yes", it 
# is wrong most of the time, as there are 37104 "no(s)" reported.
# Thus, the model would implement false positives more often with a 
# balanced model.

# Reduced model only removed no variables.
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
# at ~84% with high recall ~83%, which displays a preferred model over
# the reduced unbalanced model, which had 63.45% balanced accuracy with
# extremely low recall at ~28%, showing how the unbalanced model is
# too conservative at claiming purchases.

# RFECV was used for both, with the balanced model not being reduced at all.
# Separate analyses must be done to determine a minimal feature set, and 
# finding segments to prioritize.

# There is no clear evidence of overfitting, as the validation performance is 
# very effective. 
