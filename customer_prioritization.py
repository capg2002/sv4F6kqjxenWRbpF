# Customer prioritization by analyzing frequentist proportions. 

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# Load data
deposit_db = pd.read_csv('term-deposit-marketing-2020.csv')

# Transform data into binary and aggregate numeric into ranges.
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

# Strata analysis to arrange by highest to lowest lift.
def strata_analysis(data, feature, target):
    overall_prop = data[target].mean()

    results = (data
               .groupby(feature, observed=True)[target]
               .agg(
                   total_customers = "size",
                   buyers = "sum",
                   purchase_rate = "mean"
               )
               .reset_index()
               )

    results = results[results["total_customers"] > 400]

    results["lift"] = results["purchase_rate"]/overall_prop

# Sort by lift and then total customers. Lift shows the comparative 
# change compared to other variable's impacts.
# Total customers could be prioritized over a lower lift. If a majority of 
# the data is in cat A and cat A has a slightly lower lift from cat B, that
# is still highly valuable.
 
    return results.sort_values(
        ["lift", "total_customers"],
        ascending = [False, False]
    )

# Run comparison for every category.

for col in cols:
    job_segments = strata_analysis(
        deposit_db,
        feature = col,
        target = "y"
    )
    print(job_segments)

# Notable results:

# Jobs:
# Students and retired folks have a much higher purchase rate compared to other
# jobs. Students are 2 times more likely to commit than otherwise, while retired
# folks have 1.45x higher likelihood.
# Notably, there were 8166 customers in management, and their retention is slightly above 
# average. This is a highly populated position that still retains a good number of purchases.
# Conversely, blue-collar workers are 22% less likely to commit than the average position,
# even though they have the highest population in the dataset. Less emphasis
# should be placed on ensuring blue-collar workers commit. 

# Marital status:
# Single folks are more likely to buy than married folks, being around 1.5x more likely
# to. This suggests larger emphasis should be placed on single folks when marketing.

# Education:
# The purchase rate was generally low for all education categories.
# People with tertiary education were 1.26x more likely to commit compared to your
# average person. Primary education was 23% less likely to commit.

# Default:
# If they don't have credit in default (which is a majority of the people in the sample),
# they buy at an average rate. If it is in default, it is 17% lower.

# Housing:
# Folks with housing had roughly the same number of buyers as folks without housing,
# even though there are around 8000 more customers with housing than not. 
# Folks with housing have are 1.23x more likely to buy, while without housing, 
# it is 16% lower than average.

# Loan:
# This variable did not seem to impact things significantly. Folks without a loan are
# around 25% less likely to purchase than average.

# Month:
# Folks contacted in April have a 2.29x higher likelihood to purchase, which is 
# exceedingly high. February is also high, at 1.53x higher.
# All other months fall under the average, with January and August being the worst
# performing months, being 56% and 24% lower respectively.

# Campaign:
# Most buyers buy on their first contact, while the third contact is the second most effective,
# only being ~2% less likely to purchase. Once individuals are contacted 6 times or more,
# This drops significantly, to over 23% less likely to purchase.
# This suggests campaigns should be kept to a maximum of 5 contacts, as there are 
# little returns thereafter.

# Age group:
# 21 - 25 year olds have the highest life, being 1.86x more likely to buy.
# 26 - 30 year olds have 1.33% more likelihood.
# Seemingly, 21 - 35 year olds should be the main target, as they each have a 
# higher likelihood than average. 

# Day group:
# The highest likelihood of people paying is within the first and last weeks of the month.
# Notably, although the third week had the most customers represented, it had the lowest lift.
# Yet, all in all, the day in which they're reached out to doesn't make a huge difference, as
# there is only a 20% difference in lift from 1.13x more likely to purchase vs 7% less likely 
# to purchase.

# Bivariate comparison for every tuple of variables.
for col_1 in cols:
    for col_2 in cols:
        if (col_1 == "y") | (col_2 == "y"):
            print("skip")
        else:
            if col_1 == col_2:
                print("skip")
                continue

            job_segments = strata_analysis(
                    deposit_db,
                    feature = [col_1, col_2],
                    target = "y"
                )
            job_segments_1 = job_segments[job_segments["lift"] > 2]
            job_segments_2 = job_segments[job_segments["lift"] < 0.51]
            if (job_segments_1.empty) & (job_segments_2.empty):
                print("skip")
            elif (not job_segments_1.empty) & (job_segments_2.empty):
                print(job_segments_1)
            elif (job_segments_1.empty) & (not job_segments_2.empty):
                print(job_segments_2)
            else:
                print(job_segments_1)
                print(job_segments_2)

# Notable interactions include:
# Single students, and students with no default nor loan all are more than twice likely
# to buy. Students who are single also have a higher likelihood. 

# Interestingly, those in management in April are 3.15x more likely to buy. 

# People with no housing in April have a 5.4x higher likelihood to purchase, alluding to
# high dependence. 

# Married folks in January are 80% less likely to purchase, yielding the absolute lowest
# lift out of all bivariate interactions. This was with a sample of 570 customers,
# so the sample size is considerable too. 

# Other interactions were as consistent with independently high purchase rate
# variables. 


# There does not seem to be significant interaction between age group and education,
# with the highest lift being acquired from the same categories as previous stated.

# A naive, independently-assumed interpretation is that
# the prioritized customers should be:

# Students, single folks, people with tertiary education, people with no credit
# in default, people with housing, people contacted in April or February,
# people contacted less than 5 times, and people within 21 and 35 years of age.

# Additionally, there is very little bivariate dependence that increases lift significantly,
# aside from those in management reached out to in April.

# Further analysis can be conducted regarding interaction terms.


