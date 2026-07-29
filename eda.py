### Exploratory data analysis, ensuring a lack of missing values and 
# necessary manipulation for initial logistic regression. 

import pandas as pd

# Loading file

deposit_db = pd.read_csv('term-deposit-marketing-2020.csv')

feature_cols = deposit_db.columns

for var in feature_cols:
    variable_counts = deposit_db[[var]].value_counts()
    variable_distinct = deposit_db[[var]].nunique()
    print("Total unique categories of", var, "is", variable_distinct)
    print(variable_counts)

# Note that the response variable is greatly unbalanced,
# with a 205:16 ratio of "no" to "yes" answers.

# Must convert categories into factors.

# Also notable:
# All numeric values are integers.
# All categories are distinctly defined, and do not need to be 
# aggregated. Categorical variables must be transformed into factors variables.

print(deposit_db.nunique())
print(deposit_db.isna().value_counts())

# All binary variables are genuinely binary, with no reporting
# variation. "y" var as 0-1 numeric for logistic regression. 
# Engineering is easy since they all have yes and no as their string rep.

# There are no missing values in categorical reporting or
# in the form of NAs.

# The biggest determining factor is that the response variable is
# imbalanced (205:16 split), which must be accounted for by looking at balanced
# accuracy instead of raw accuracy. 