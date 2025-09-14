import pandas as pd
import matplotlib.pyplot as plt

# load dataset
df = pd.read_csv('/Users/urvashibalasubramaniam/Documents/GitHub/Kabir_Vohra_Urvashi_Balasubraniam_A1/retail_task/src/Retail.csv')

################

# DATE FIELD HANDLING
# converting numerical to datetime datatype (for interval calculation)
df['promotion_start_date'] = pd.to_datetime(df['promotion_start_date'])
df['promotion_end_date'] = pd.to_datetime(df['promotion_end_date'])

# creating new column in df = interval b/w promotion start and end date (unit = days)
df['promotion_time_period'] = (df['promotion_end_date'] - df['promotion_start_date']).dt.days

# dropping (deleting) date columns in new dataframe
df_new = df.drop(['transaction_date','last_purchase_date','product_manufacture_date','product_expiry_date','promotion_start_date','promotion_end_date'], axis=1)

# overwrite df with df_new
df = df_new

################

# ONE HOT ENCODING

# getting category cols, from overwritten (just modified) df
categoryCols = df.select_dtypes(exclude=["number"]).columns.tolist()

df_encoded = pd.get_dummies(df, columns=categoryCols)

df = df_encoded # overwriting as it is correctly doing the operation

onehotcols = df.select_dtypes(include=['bool']).columns # all one hot encoded cols (have boolean values)

# typecasting all bools to integers (False -> 0, True -> 1)
df[onehotcols] = df[onehotcols].astype(int) 

################
# NUMERICAL COLS (after one hot encoding, so including transformed categorical data)
numericalCols = df.select_dtypes(include=["number"]).columns.tolist()

# NORMALISATION (of training)
# included initially, but using z score normalisation instead directly in train_model.py
# loop through each numerical column & apply Min-Max scaling
for col in numericalCols:
    min_val = df[col].min()
    max_val = df[col].max()
    # avoid division by zero! -- happens when all values in the column are the same
    if max_val - min_val != 0:
        df[col] = (df[col] - min_val) / (max_val - min_val)
    else:
        # if all values are the same, the normalized value is 0
        df[col] = 0

##############################

# TRAINING DATA SPECIFICATION
# using data_exploration.ipnyb's best correlated features to train the model
"""
NUMERICAL FEATURES:
total_returned_items        0.002510
in_store_purchases          0.002482
quantity                    0.002272
transaction_id              0.001726
total_items_purchased       0.001661
website_visits              0.001527
customer_zip_code           0.001292
days_since_last_purchase    0.000979
product_review_count        0.000955

they're all weakly correlated^ so exclusion of data isn't helpful

CATEGORICAL FEATURES:
gender_Male               0.002337
customer_city_City C      0.002578
customer_state_State Z   -0.002529
store_state_State Z      -0.002281

OVERALL FEATURES:
1. avg_purchase_value       1.000000
2. customer_city_City C     0.002578
3. total_returned_items     0.002510
4. in_store_purchases       0.002482
5. gender_Male              0.002337
6. quantity                 0.002272
7. payment_method_Cash      0.002060
8. transaction_id           0.001726
9. total_items_purchased    0.001661
10. loyalty_program_No       0.001652

"""

# initially exporting the file as usual
df.to_csv('training_data_1.csv', index=False)

# selected columns only
selected_columns = ['avg_purchase_value', 'customer_city_City C', 'total_returned_items','in_store_purchases','gender_Male','quantity','payment_method_Cash','transaction_id','total_items_purchased','website_visits','customer_zip_code','days_since_last_purchase','product_review_count','store_state_State Z','customer_state_State Z']
df_subset = df[selected_columns]
df_subset.to_csv('training_subset.csv', index=False)