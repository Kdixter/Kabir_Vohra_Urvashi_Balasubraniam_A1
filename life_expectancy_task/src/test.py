import pandas as pd
import numpy as np
import pickle
import os
import argparse

# To load the saved .pkl model, we need the class definition.
# We import it directly from the training script.
from train_model import LinearRegressionScratch

def evaluate_model(model_path, data_path, metrics_output_path, predictions_output_path):
    """
    Loads a trained model, evaluates it on test data, and saves the
    predictions and performance metrics.
    """
    try:
        # --- 1. LOAD THE TRAINED MODEL AND TEST DATA ---
        print("Loading trained model and test data...")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        test_df = pd.read_csv(data_path)
        
        # Also load the original training data to get scaling parameters (min/max)
        # This is CRITICAL to ensure the test data is scaled the same way as the training data.
        training_data_path = os.path.join(os.path.dirname(data_path), 'training_set.csv')
        train_df = pd.read_csv(training_data_path)


    except FileNotFoundError as e:
        print(f"Error: A required file was not found. {e}")
        print("Please ensure the model is trained and all data files are in the 'data' directory.")
        return
    except Exception as e:
        print(f"An error occurred during file loading: {e}")
        return

    # --- 2. PREPARE THE TEST DATA ---
    print("Preparing test data...")
    # Separate features (X_test) from the actual target values (y_test)
    X_test = test_df.drop(['Country', 'Life expectancy '], axis=1)
    y_test = test_df['Life expectancy ']

    # Prepare the original training features to get min/max for scaling
    X_train = train_df.drop(['Country', 'Life expectancy '], axis=1)
    
    # Scale the test data using the MIN and MAX values from the TRAINING data.
    # This prevents any "data leakage" from the test set into the model.
    X_test_scaled = (X_test - X_train.min()) / (X_train.max() - X_train.min())

    # --- 3. MAKE PREDICTIONS ---
    print("Making predictions on the test set...")
    predictions = model.predict(X_test_scaled)

    # --- 4. CALCULATE PERFORMANCE METRICS FROM SCRATCH ---
    print("Calculating performance metrics...")
    
    # Mean Squared Error (MSE)
    mse = np.mean((y_test - predictions) ** 2)

    # Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)

    # R-squared (R²) Score
    # R² = 1 - (Sum of Squared Residuals / Total Sum of Squares)
    ss_res = np.sum((y_test - predictions) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2_score = 1 - (ss_res / ss_tot)
    
    print(f" - MSE: {mse:.2f}")
    print(f" - RMSE: {rmse:.2f}")
    print(f" - R-squared: {r2_score:.2f}")

    # --- 5. SAVE THE RESULTS ---
    
    # Save the predictions to a CSV file
    pd.DataFrame(predictions, columns=['predictions']).to_csv(predictions_output_path, index=False, header=False)
    print(f"\nPredictions saved to: {predictions_output_path}")

    # Save the metrics to a text file in the required format
    with open(metrics_output_path, 'w') as f:
        f.write("Regression Metrics:\n")
        f.write(f"Mean Squared Error (MSE): {mse:.2f}\n")
        f.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}\n")
        f.write(f"R-squared (R²) Score: {r2_score:.2f}\n")
    print(f"Metrics saved to: {metrics_output_path}")


if __name__ == '__main__':
    # This block allows the script to be run from the command line
    # with arguments, as specified in the assignment instructions.
    parser = argparse.ArgumentParser(description="Evaluate a trained regression model.")
    
    # Set up default paths assuming the script is run from the project root
    default_model_path = 'life_expectancy_task/models/regression_model1.pkl'
    default_data_path = 'life_expectancy_task/data/testing_set.csv'
    default_metrics_path = 'life_expectancy_task/results/train_metrics.txt' # As per assignment naming
    default_preds_path = 'life_expectancy_task/results/train_predictions.csv' # As per assignment naming

    parser.add_argument('--model_path', type=str, default=default_model_path, help='Path to the saved model file.')
    parser.add_argument('--data_path', type=str, default=default_data_path, help='Path to the test data CSV file.')
    parser.add_argument('--metrics_output_path', type=str, default=default_metrics_path, help='Path to save the evaluation metrics.')
    parser.add_argument('--predictions_output_path', type=str, default=default_preds_path, help='Path to save the predictions.')
    
    args = parser.parse_args()
    
    evaluate_model(args.model_path, args.data_path, args.metrics_output_path, args.predictions_output_path)