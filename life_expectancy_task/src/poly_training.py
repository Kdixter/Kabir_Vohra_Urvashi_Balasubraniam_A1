import numpy as np
import pandas as pd
import os
import pickle

# We can reuse the same linear regression class.
from train_model import LinearRegressionScratch

# This block allows you to run the script directly to train the final model
if __name__ == "__main__":
    print("--- Starting Final Model Training ---")
    
    # 1. Load the final, SCALED training data
    try:
        scaled_data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'poly_training_set_scaled.csv')
        df = pd.read_csv(scaled_data_path)
        print(f"Successfully loaded scaled training data from: {scaled_data_path}")
    except FileNotFoundError:
        print("Scaled data file not found. Please run the entire data pipeline (preprocess, split, engineer, scale) first.")
        exit()

    # 2. Prepare the data for the model
    # Drop the non-numeric 'Country' column
    df_numeric = df.drop('Country', axis=1)

    # Separate features (X) from the target (y)
    # Note: The data is already scaled, so we use it directly.
    X_final = df_numeric.drop('Life expectancy ', axis=1)
    y = df_numeric['Life expectancy ']

    # !!! CRITICAL: We DO NOT apply scaling again here !!!
    print("Using pre-scaled data directly for training.")
    print(f"Training with {len(X_final.columns)} features.")

    # 3. Train the Linear Regression model on the final features
    # A smaller learning rate can help with stability for complex models.
    final_model = LinearRegressionScratch(learning_rate=0.005, n_iterations=5000)
    final_model.fit(X_final, y)

    # 4. Display the learned weights
    print("\n--- Final Model Training Complete ---")
    
    # Check if training was successful (weights are not NaN)
    if np.isnan(final_model.bias):
        print("\nERROR: Training failed. Weights are NaN. Try a smaller learning rate.")
    else:
        print(f"Bias (Intercept): {final_model.bias:.4f}")
        weights_df = pd.DataFrame({
            'Feature': X_final.columns,
            'Weight': final_model.weights
        }).sort_values(by='Weight', ascending=False)
        
        print("\nLearned Feature Weights for Final Model:")
        print(weights_df)

        # 5. Save the final trained model
        try:
            model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'regression_model_2.pkl')
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            with open(model_path, 'wb') as f:
                pickle.dump(final_model, f)
            print(f"\nFinal model successfully saved to: {model_path}")
        except Exception as e:
            print(f"\nError saving the model: {e}")
