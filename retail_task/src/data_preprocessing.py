import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# dependent variable (what we are predicting)
# = avg_purchase_value per customer

# y = m1 x1 + m2 x2 + ... + m77 x77

data = pd.read_csv('/Users/urvashibalasubramaniam/Documents/GitHub/Kabir_Vohra_Urvashi_Balasubraniam_A1/retail_task/src/Retail.csv')
print(data)

# mean squared error
def mse(m,b, points):
    total_error = 0 # for summation
    for i in range(len(points)):
        x = points.iloc()

# root mean squared error

# r squared (r^2) score

# round values to 2 decimal places