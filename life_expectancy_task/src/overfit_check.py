import pandas as pd
import numpy as np
import pickle
import os
import argparse

# We need the class definition to load the model
from train_model import LinearRegressionScratch

def check_training_performance(model_path, training_data_path):
    """
    Loads a trained model and evaluates its performance on the data it was
    trained on to help diagnose overfitting.
    """
    try:
        # --- 1. LOAD THE TRAINED MODEL AND TRAINING DATA ---
        print(f"Loading model from: {model_path}")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        print(f"Loading the scaled training data from: {training_data_path}")
        train_df = pd.read_csv(training_data_path)

    except FileNotFoundError as e:
        print(f"Error: A required file was not found. {e}")
        return

    # --- 2. PREPARE THE DATA ---
    # The data is already scaled, so we just need to separate X and y
    X_train = train_df.drop(['Country', 'Life expectancy '], axis=1)
    y_train = train_df['Life expectancy ']

    # --- 3. MAKE PREDICTIONS ON THE TRAINING DATA ---
    print("Making predictions on the training set...")
    predictions = model.predict(X_train)

    # --- 4. CALCULATE PERFORMANCE METRICS ---
    print("\n--- Performance on TRAINING Data ---")
    mse = np.mean((y_train - predictions) ** 2)
    rmse = np.sqrt(mse)
    ss_res = np.sum((y_train - predictions) ** 2)
    ss_tot = np.sum((y_train - np.mean(y_train)) ** 2)
    r2_score = 1 - (ss_res / ss_tot)
    
    print(f" - MSE: {mse:.2f}")
    print(f" - RMSE: {rmse:.2f}")
    print(f" - R-squared: {r2_score:.2f}")
    print("------------------------------------")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Check a model's performance on its training data.")
    
    # Point this to the model you want to check
    default_model_path = 'life_expectancy_task/models/regression_model_2.pkl'
    # Point this to the data that model was trained on
    default_data_path = 'life_expectancy_task/data/training_set_scaled.csv'

    parser.add_argument('--model_path', type=str, default=default_model_path)
    parser.add_argument('--data_path', type=str, default=default_data_path)
    
    args = parser.parse_args()
    
    check_training_performance(args.model_path, args.data_path)