# Without duration, elastic net, pure logistic regression, histogram gradient boosting,
# and random forest all yielded balanced accuracies of around 63%, suggesting the variables,
# without duration, only hold middling predictive power.

# This file is meant to show all these alternative attempts. 
# Following the full analysis, and, in an effort to not overfit, this shows the lower
# predictive power.

import pandas as pd
import numpy as np
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold,
    train_test_split,
    cross_validate,
)
from sklearn.model_selection import (
    FixedThresholdClassifier,
    TunedThresholdClassifierCV,
    cross_validate
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

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              RandomForestClassifier)

warnings.filterwarnings('ignore')
# Note 15000 iterations was tested, and the results were identical to 
# having 5000 iterations.
seed = 23
iterations = 2000

# Loading data
deposit_db = pd.read_csv('term-deposit-marketing-2020.csv')

# Transforming data for fitting, as explained in initial_logistic_regression.py
bin_cols = deposit_db.columns[deposit_db.nunique() == 2]
deposit_db[bin_cols] = deposit_db[bin_cols].replace({'yes': 1, 'no': 0})

factor_cols = deposit_db.select_dtypes(include = ['object', 'category'])
factor_cols = factor_cols.columns[factor_cols.nunique() > 2]

deposit_db = pd.get_dummies(
    deposit_db,
    columns=factor_cols,
    drop_first = True,
    dtype=int)

Y_var = deposit_db["y"]
X_full = deposit_db.drop(["y", "duration"], axis=1)

cv = StratifiedKFold(n_splits=5,
    shuffle=True,
    random_state=seed
    )

# Train-test split
X_train_f, X_test_f, Y_train, Y_test = train_test_split(X_full, 
            Y_var, test_size=0.2, random_state=seed)

# Full unbalanced model.
full_model = LogisticRegression(max_iter = iterations)
full_model.fit(X_train_f, Y_train)

scores_full = cross_validate(
    full_model,
    X_train_f,
    Y_train,
    cv=cv,
    scoring=["balanced_accuracy", "accuracy", "precision", "recall", 
             "f1", "roc_auc", "neg_log_loss"]
)

# Printing diagnostics
scores_full_names = list(scores_full.keys())
for name in scores_full_names:
    print("Full", name, ":", scores_full[name].mean())

# As expected, an accuracy of 92.80% and a balanced accuracy of 51.59%, this
# model greatly underperforms. 


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
    print("Full balanced", name, ":", scores_full[name].mean())

print("FINAL TEST RESULTS")
print("Accuracy:", accuracy_score(Y_test, Y_pred_f))
print("Balanced accuracy:", balanced_accuracy_score(Y_test, Y_pred_f))
print("Precision:", precision_score(Y_test, Y_pred_f))
print("Recall:", recall_score(Y_test, Y_pred_f))
print("F1:", f1_score(Y_test, Y_pred_f))
print("ROC AUC:", roc_auc_score(Y_test, Y_prbs_f))
print("Log loss:", log_loss(Y_test, Y_prbs_f))

# Compared to the original model, this model underperforms significantly,
# only having a balanced accuracy of 61.13%. More models should be tested.


# Attempting Histogram Gradient Boosting; a nonlinear optimization
# algorithm to identify additional trends. This model is 
# harder to interpret, but may yield a high balanced accuracy.

# Best histogram gradient boosting model found previously.

threshold = 0.08
boosting_model = HistGradientBoostingClassifier(
    class_weight=None,
    learning_rate=0.1,
    max_iter=300,
    max_leaf_nodes=15,
    min_samples_leaf=100,
    l2_regularization=0,
    random_state=seed
)

chosen_model = FixedThresholdClassifier(
    estimator=boosting_model,
    threshold=threshold,
    response_method="predict_proba"
)

tuned_boosting_scores = cross_validate(
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
        "average_precision",
        "neg_log_loss"
    ],
    n_jobs=1,
    return_estimator=True,
    error_score="raise"
)

for name, values in tuned_boosting_scores.items():
    if name == "estimator":
        continue
    else:
        print(
            name,
            ":",
            values.mean()
        )

chosen_model.fit(X_train_f, Y_train)

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


# This improved the balanced accuracy by roughly 2% to 63.46%, which is marginal,
# but it is a minor improvement.

forest_model = RandomForestClassifier(
    n_estimators=500,
    max_features="sqrt",
    min_samples_leaf=10,
    class_weight="balanced_subsample",
    random_state=seed,
    n_jobs=-1
)

forest_scores = cross_validate(
    forest_model,
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
        "average_precision",
        "neg_log_loss"
    ]
)

for name, values in forest_scores.items():
    print(name, ":", values.mean())

forest_model.fit(X_train_f, Y_train)

Y_pred_f = forest_model.predict(X_test_f)
Y_prbs_f = forest_model.predict_proba(X_test_f)[:,1]


print("FINAL TEST RESULTS")
print("Accuracy:", accuracy_score(Y_test, Y_pred_f))
print("Balanced accuracy:", balanced_accuracy_score(Y_test, Y_pred_f))
print("Precision:", precision_score(Y_test, Y_pred_f))
print("Recall:", recall_score(Y_test, Y_pred_f))
print("F1:", f1_score(Y_test, Y_pred_f))
print("ROC AUC:", roc_auc_score(Y_test, Y_prbs_f))
print("Log loss:", log_loss(Y_test, Y_prbs_f))

# The random forest classifier is the worst out of all these models. It has the lowest
# balanced accuracy, precision and recall. With that being said, it has the 
# highest accuracy out of all of the models. 

# Notably, all of these models converge to a balanced accuracy of around 63%. 
# Through an exploration of different starting parameters, performance did not improve
# meaningfully. This further corroborates that the model's variables,
# outside of duration, do not hold significant predictive ability.

chosen_model.fit(X_train_f, Y_train)

expected_features = X_full.columns.tolist()

people_raw = pd.DataFrame([
    {
        "profile": "Single tertiary student in February",
        "age": 23,
        "job": "student",
        "marital": "single",
        "education": "tertiary",
        "default": 0,
        "balance": 1500,
        "housing": 0,
        "loan": 0,
        "contact": "cellular",
        "day": 5,
        "month": "feb",
        "duration": 124,
        "campaign": 1
    },
    {
        "profile": "Retired customer in October",
        "age": 68,
        "job": "retired",
        "marital": "married",
        "education": "secondary",
        "default": 0,
        "balance": 3500,
        "housing": 0,
        "loan": 0,
        "contact": "cellular",
        "day": 29,
        "month": "oct",
        "duration": 124,
        "campaign": 1
    },
    {
        "profile": "Management customer in April",
        "age": 38,
        "job": "management",
        "marital": "married",
        "education": "tertiary",
        "default": 0,
        "balance": 2500,
        "housing": 0,
        "loan": 0,
        "contact": "cellular",
        "day": 5,
        "month": "apr",
        "duration": 124,
        "campaign": 1
    },
    {
        "profile": "Low-priority blue-collar customer",
        "age": 48,
        "job": "blue-collar",
        "marital": "married",
        "education": "primary",
        "default": 1,
        "balance": 200,
        "housing": 1,
        "loan": 1,
        "contact": "unknown",
        "day": 16,
        "month": "jan",
        "duration": 124,
        "campaign": 7
    },
    {
        "profile": "Mixed services customer",
        "age": 29,
        "job": "services",
        "marital": "single",
        "education": "tertiary",
        "default": 0,
        "balance": 1000,
        "housing": 1,
        "loan": 0,
        "contact": "cellular",
        "day": 16,
        "month": "aug",
        "duration": 124,
        "campaign": 3
    }
])

profile_names = people_raw["profile"].copy()

# Drop profile name
X_people_raw = people_raw.drop(columns="profile")

# Making the people using dummy variables
X_people_encoded = pd.get_dummies(
    X_people_raw,
    columns=factor_cols,
    drop_first=False,
    dtype=int
)

# Make all expected features be 0 if not filled, and add potential missing columns.
X_people = X_people_encoded.reindex(
    columns=expected_features,
    fill_value=0
)

positive_class_index = list(
    chosen_model.classes_
).index(1)

purchase_probabilities = chosen_model.predict_proba(
    X_people
)[:, positive_class_index]

# Create dataframe with genuine results using model.
results = pd.DataFrame({
    "profile": profile_names,
    "purchase_probability": purchase_probabilities
})

results["purchase_probability_percent"] = (
    results["purchase_probability"] * 100
).round(2)

results = results.sort_values(
    "purchase_probability",
    ascending=False
)

print(results)

results["predicted_buyer"] = (
    results["purchase_probability"] >= threshold
).astype(int)

results["recommendation"] = np.where(
    results["predicted_buyer"] == 1,
    "Prioritize",
    "Do not prioritize"
)

print(
    results[
        [
            "profile",
            "purchase_probability_percent",
            "recommendation"
        ]
    ]
)

# Note the model accepts the single tertiary student in Feb and rejects 
# the blue collar customer, as expected from the customer prioritization doc.
# This is much more consistent with the frequentist analysis seen in the 
# customer_prioritization.py doc compared to the model with duration.

# If regular accuracy is prioritized, the random forest should be used.

# If balanced accuracy is prioritized then the histogram gradient model
# should be prioritized. For the purpose of prediction, this model should be used.