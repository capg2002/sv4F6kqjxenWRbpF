import pandas as pd
import numpy as np
import textwrap
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    train_test_split,
    cross_val_score,
    cross_validate,
    GridSearchCV,
)

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV

from sklearn.feature_selection import SequentialFeatureSelector

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


chosen_C = 0.003162

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