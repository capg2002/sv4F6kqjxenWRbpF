import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import (
    StratifiedKFold,
    train_test_split,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LogisticRegression


warnings.filterwarnings("ignore")

seed = 22
iterations = 2000

# Loading data
deposit_db = pd.read_csv('term-deposit-marketing-2020.csv')

# Transforming data for fitting, as explained in initial_logistic_regression.py
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

Y_var = deposit_db["y"]
X_full = deposit_db.drop(["y", "duration"], axis=1)

cv = StratifiedKFold(n_splits=5,
    shuffle=True,
    random_state=seed
    )

# Train-test split
X_train_f, X_test_f, Y_train, Y_test = train_test_split(X_full, 
            Y_var, test_size=0.2, random_state=seed)

interaction_model = Pipeline([
    (
        "interactions",
        PolynomialFeatures(
            degree=2,
            interaction_only=True,
            include_bias=False
        )
    ),
    (
        "scaler",
        StandardScaler()
    ),
    (
        "logistic",
        LogisticRegression(
            max_iter=2000,
            penalty = "l1",
            solver = "liblinear",
            class_weight = "balanced",
            C = 0.00329,
            random_state=seed
        )
    )
])

interaction_scores = cross_validate(
    interaction_model,
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

# Print mean cross-validation diagnostics
for name, values in interaction_scores.items():
    print(name, ":", values.mean())

# cross_validate fits copies of the model, so fit the original model
# on the complete training dataset before accessing coefficients.
interaction_model.fit(X_train_f, Y_train)

# Extract fitted coefficients
coefficients = interaction_model.named_steps["logistic"].coef_[0]

# Get the names of the original and interaction variables
interaction_names = (
    interaction_model
    .named_steps["interactions"]
    .get_feature_names_out(X_train_f.columns)
)

# Keep variables with non-zero L1 coefficients
selected_features = interaction_names[
    np.abs(coefficients) > 1e-8
]

print("Selected features:")
print(selected_features.tolist())

print(len(selected_features.tolist()))

# This model has a significant number of features, being 79 selected variables,
# but the balanced accuracy only marginally improved from the model without
# interaction terms. Evidently, this supports the concept that there is only
# middling predictive power within these variables.

