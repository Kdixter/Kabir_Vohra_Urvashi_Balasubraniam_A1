import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing as MinMaxScaler
from sklearn.compose import ColumnTransformer

df = pd.read_csv('/Users/urvashibalasubramaniam/Documents/GitHub/Kabir_Vohra_Urvashi_Balasubraniam_A1/retail_task/src/Retail.csv')

""" 
Refer data_exploration.ipynb for checks 
 for missing values, var types, statistical distributions etc """

# Separating numerical and categorical (non-numerical/string) data value columns
numericalCols = df.select_dtypes(include=["number"]).columns.tolist()
categoryCols = df.select_dtypes(exclude=["number"]).columns.tolist()

# One hot encoding using scikitlearn
ct = ColumnTransformer( transformers=[
        ("cat", MinMaxScaler.OneHotEncoder(handle_unknown='ignore'), categoryCols),
        ("num", "passthrough", numericalCols) # passthrough = continue 
    ])
"""
Transformation: 
cat (transform name) -> apply OneHotEncoder to categoryCols
num (transform name)-> just pass through without applying anything
Called later using ct.fit_transform(df) which outputs the numpy array (without headers)
"""

# Call the above transform (outputs numpy array)
tempEncoded = ct.fit_transform(df)

# Add back the headers and numerical data
cat_features = ct.named_transformers_["cat"].get_feature_names_out(categoryCols)
all_features = list(cat_features) + numericalCols

# Making it a dataframe again
df_encoded = pd.DataFrame(tempEncoded.toarray(), columns=all_features)

print(df_encoded.head())

# Correlation analysis between target and all other variables

# Normalisation (MinMax Scaling) -- try and do this without scikitlearn

# Standardisation
"""
Transforms features to have mean = 0 and standard deviation = 1, 
useful for normally distributed features."""

# Detect and remove outliers