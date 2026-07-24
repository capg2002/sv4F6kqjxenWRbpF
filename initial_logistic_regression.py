# Initial Logistic regression attempt

import pandas as pd
import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold,
    train_test_split,
    cross_validate,
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
bin_cols = deposit_db.columns[deposit_db.nunique() == 2]
deposit_db[bin_cols] = deposit_db[bin_cols].replace({'yes': 1, 'no': 0})

# Transforming cat columns into factors.
factor_cols = deposit_db.select_dtypes(include = ['object', 'category'])
factor_cols = factor_cols.columns[factor_cols.nunique() > 2]

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
    X_full,
    Y_var,
    cv=cv,
    scoring=["balanced_accuracy", "accuracy", "precision", "recall", 
             "f1", "roc_auc", "neg_log_loss"]
)

# Printing diagnostics
scores_full_names = list(scores_full.keys())
for name in scores_full_names:
    print("Full", name, ":", scores_full[name].mean())

# Accuracy is high, at 93.5%, but the balanced accuracy is 63.42%.
# Note that 92.76% of the dataset is negatives, so this likely predicts
# the majority of the negatives correctly, but also predicts false negatives often. 

# This is corroborated by the recall at 28.24%, (TP/(TP + FN)) showing how a 
# the model misses actual positive cases, resulting in a high number of false negatives.

# This suggests that the model over-relies on predicting negatives.

# RFECV selection did not remove any variables. Thus, the model is the same.
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
    X_full,
    Y_var,
    cv=cv,
    scoring=["balanced_accuracy", "accuracy", "precision", "recall", "f1", "roc_auc", "neg_log_loss"]
)

scores_full_names = list(scores_full.keys())
for name in scores_full_names:
    print("Full", name, ":", scores_full[name].mean())

# Notably, this full balanced model is considerably more consistent, improving
# balanced accuracy and recall significantly. 

# Balanced accuracy is 86.42% and recall is 85.97%, which is very strong.
# Notably, accuracy and precision have decreasing. It makes sense that 
# precision decreases, as it shows that, when the model says "yes", it 
# is wrong most of the time, as there are 37104 "no(s)" reported.
# Thus, the model would implement false positives more often with a 
# balanced model.

# Reduced model only removed one variable, which increased balanced accuracy marginally.
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

# There is no clear evidence of overfitting, as the validation performance is 
# very effective. 
