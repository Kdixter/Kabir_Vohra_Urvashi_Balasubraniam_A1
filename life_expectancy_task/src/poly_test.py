import pandas as pd
import numpy as np
import pickle
import os
import argparse
from train_model import LinearRegressionScratch # Needed to load the model

def evaluate_final_model(model_path, data_path, metrics_output_path, predictions_output_path):
    """
    Loads the final trained model and evaluates it on the pre-processed test set.
    """
    try:
        # 1. Load the model and the pre-processed test data
        print("Loading trained model and data files...")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        test_df = pd.read_csv(data_path)

    except FileNotFoundError as e:
        print(f"Error: A required file was not found. {e}")
        print("Please ensure you have run run_data_pipeline.py and train_final_model.py first.")
        return
    except Exception as e:
        print(f"An error occurred during file loading: {e}")
        return

    # 2. Prepare the test data (it's already clean and scaled)
    # The number of features here will now match the model's weights.
    X_test = test_df.drop(columns=['Country', 'Life expectancy '])
    y_test = test_df['Life expectancy ']

    # 3. Make predictions
    print("Making predictions on the test set...")
    predictions = model.predict(X_test)

    # 4. Calculate and save metrics
    print("Calculating performance metrics...")
    mse = np.mean((y_test - predictions) ** 2)
    rmse = np.sqrt(mse)
    ss_res = np.sum((y_test - predictions) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2_score = 1 - (ss_res / ss_tot)
    
    print(f" - MSE: {mse:.2f}, RMSE: {rmse:.2f}, R-squared: {r2_score:.2f}")

    pd.DataFrame(predictions).to_csv(predictions_output_path, index=False, header=False)
    with open(metrics_output_path, 'w') as f:
        f.write("Regression Metrics:\n")
        f.write(f"Mean Squared Error (MSE): {mse:.2f}\n")
        f.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}\n")
        f.write(f"R-squared (R²) Score: {r2_score:.2f}\n")
    print(f"\nResults saved successfully.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate the final trained model.")
    default_model_path = 'life_expectancy_task/models/regression_model_2.pkl'
    # Updated to use the training data for testing, as requested.
    default_data_path = 'life_expectancy_task/data/poly_train.csv'
    default_metrics_path = 'life_expectancy_task/results/train_metrics.txt'
    default_preds_path = 'life_expectancy_task/results/train_predictions.csv'

    parser.add_argument('--model_path', type=str, default=default_model_path)
    parser.add_argument('--data_path', type=str, default=default_data_path)
    parser.add_argument('--metrics_output_path', type=str, default=default_metrics_path)
    parser.add_argument('--predictions_output_path', type=str, default=default_preds_path)
    
    args = parser.parse_args()
    
    evaluate_final_model(args.model_path, args.data_path, args.metrics_output_path, args.predictions_output_path)

