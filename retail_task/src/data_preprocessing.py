import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing as MinMaxScaler

df = pd.read_csv('/Users/urvashibalasubramaniam/Documents/GitHub/Kabir_Vohra_Urvashi_Balasubraniam_A1/retail_task/src/Retail.csv')

""" 
Refer data_exploration.ipynb for checks 
 for missing values, var types, statistical distributions etc """

# Correlation analysis

"""
Need to perform one-hot encoding on the following categorical variables. Of the 77 columns:
(1) Identifying those that have Strings as fields
(2) Coming up with a standard rule to perform one-hot encoding without my manual intervention required.

For pandas, we usually use
uniqueCount = data['category_name'].nunique()

(3) Coming up with a standard rule to perform one-hot encoding without my manual intervention required.

print(f"The number of distinct product IDs is: {unique_products}")

"""