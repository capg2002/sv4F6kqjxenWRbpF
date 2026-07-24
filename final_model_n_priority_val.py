# Final chosen model

import pandas as pd
import numpy as np
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold,
    train_test_split,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
# Note 15000 iterations was tested, and the results were identical to 
# having 5000 iterations.
seed = 22
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
X_full = deposit_db.drop("y", axis=1)

cv = StratifiedKFold(n_splits=5,
    shuffle=True,
    random_state=seed
    )

# Train-test split
X_train_f, X_test_f, Y_train, Y_test = train_test_split(X_full, 
            Y_var, test_size=0.2, random_state=seed)

# Same process as in feature_selection.py
chosen_C = 0.00329

chosen_model = Pipeline([
    ("scaler", StandardScaler()),
    ("logistic", LogisticRegression(
        penalty="l1",
        solver="liblinear",
        C=chosen_C,
        class_weight=None,
        max_iter=5000
    ))
])

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

for name, values in chosen_scores.items():
    print(name, ":", values.mean())

chosen_model.fit(X_train_f, Y_train)

coefficients = chosen_model.named_steps["logistic"].coef_[0]

selected_features = X_train_f.columns[
    np.abs(coefficients) > 1e-8
]

print("Selected features:")
print(selected_features.tolist())


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

for name, values in chosen_scores.items():
    print(name, ":", values.mean())

chosen_model.fit(X_train_f, Y_train)

coefficients = chosen_model.named_steps["logistic"].coef_[0]

selected_features = X_train_f.columns[
    np.abs(coefficients) > 1e-8
]

print("Selected features:")
print(selected_features.tolist())

# The final chosen model has 85.58% balanaced accuracy, and is chosen for its feature
# simplicity compared to the full model. Its recall is 84.04%, which is significantly
# better than the original unbalanced model. 

# All columns passed into chosen_model.fit()
expected_features = chosen_model.feature_names_in_.tolist()

print("Number of model inputs:", len(expected_features))
print(expected_features)

# Below is a sample of some of the people whose likelihood can be predicted using
# the model. Duration is kept constant as the modal duration reported.

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

threshold = 0.50

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
# the blue collar customer, as expected. 