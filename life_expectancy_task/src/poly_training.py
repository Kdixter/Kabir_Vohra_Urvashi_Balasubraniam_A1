# train a polynomial regression model (degree 2) using the engineered features: 
import numpy as np
import pandas as pd
import os
import pickle

# We can reuse the same linear regression class because polynomial regression
# is just linear regression on an expanded set of features.
from train_model import LinearRegressionScratch

def generate_polynomial_features(df, degree=2):
    """
    Generates polynomial features of a specified degree.
    For this project, we will focus on degree 2 (squared terms).

    Args:
        df (pd.DataFrame): The input dataframe with the original features.
        degree (int): The degree of the polynomial (default is 2).

    Returns:
        pd.DataFrame: A new dataframe with the original and polynomial features.
    """
    df_poly = df.copy()
    
    # Get only the numeric columns to create polynomial features from
    numeric_cols = df.select_dtypes(include=np.number).columns
    
    if degree == 2:
        for col in numeric_cols:
            # Add the squared term for each numeric feature
            df_poly[f'{col}^2'] = df[col] ** 2
            
    # Higher degrees can be added here if needed
    
    return df_poly


# This block allows you to run the script directly to train the model
if __name__ == "__main__":
    print("--- Starting Polynomial Model Training (Degree 2) ---")
    
    # 1. Load the engineered training data
    try:
        engineered_data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'polynomial_training_set_engineered.csv')
        df = pd.read_csv(engineered_data_path)
        print(f"Successfully loaded engineered data from: {engineered_data_path}")
    except FileNotFoundError:
        print("Engineered data file not found. Please run engineer_features.py first.")
        exit()

    # 2. Prepare the data for the model
    # Drop the non-numeric 'Country' column
    df_numeric = df.drop('Country', axis=1)

    # Separate features (X) from the target (y)
    X = df_numeric.drop('Life expectancy ', axis=1)
    y = df_numeric['Life expectancy ']

    # 3. Generate Degree-2 Polynomial Features (Squared Terms)
    print("Generating degree-2 features (squared terms)...")
    X_poly = generate_polynomial_features(X, degree=2)
    print(f"Original number of features: {len(X.columns)}")
    print(f"New number of features after adding squared terms: {len(X_poly.columns)}")

    # 4. Feature Scaling (Very Important for Polynomial Models)
    # Scale all features (original, interactions, and squared) to be between 0 and 1.
    X_poly_scaled = (X_poly - X_poly.min()) / (X_poly.max() - X_poly.min())

    # 5. Train the Linear Regression model on the new polynomial features
    # We might need more iterations or a smaller learning rate for more complex models.
    poly_model = LinearRegressionScratch(learning_rate=0.01, n_iterations=3000)
    poly_model.fit(X_poly_scaled, y)

    # 6. Display the learned weights
    print("\n--- Polynomial Model Training Complete ---")
    print(f"Bias (Intercept): {poly_model.bias:.4f}")
    
    weights_df = pd.DataFrame({
        'Feature': X_poly.columns,
        'Weight': poly_model.weights
    }).sort_values(by='Weight', ascending=False)
    
    print("\nLearned Feature Weights for Polynomial Model:")
    print(weights_df)

    # 7. Save the trained polynomial model to a new .pkl file
    try:
        model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'regression_model2.pkl')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(poly_model, f)
        print(f"\nPolynomial model successfully saved to: {model_path}")
    except Exception as e:
        print(f"\nError saving the model: {e}")
