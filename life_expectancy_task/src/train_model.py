# implementing linear regression model first: 
import numpy as np
import pandas as pd
import os
import pickle

# This script no longer needs to import the preprocessing function,
# as it will use the pre-split training data.

class LinearRegressionScratch:
    """
    A Linear Regression model implemented from scratch using NumPy.
    
    This model uses Gradient Descent to find the optimal weights for the features.
    """
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        """
        Initializes the model with hyperparameters.

        Args:
            learning_rate (float): The step size for each iteration of gradient descent.
            n_iterations (int): The number of times to iterate through the dataset.
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Trains the linear regression model.

        Args:
            X (pd.DataFrame or np.ndarray): The training input features.
            y (pd.Series or np.ndarray): The target values.
        """
        # Convert inputs to NumPy arrays
        X = np.array(X)
        y = np.array(y)
        
        n_samples, n_features = X.shape

        # 1. Initialize the weights and bias
        # We start with small random values or zeros. Zeros are fine.
        self.weights = np.zeros(n_features)
        self.bias = 0

        # 2. Implement Gradient Descent
        for _ in range(self.n_iterations):
            # Calculate the current predictions (y_hat) using matrix multiplication
            # y_predicted = X * w + b
            y_predicted = np.dot(X, self.weights) + self.bias

            # Calculate the derivatives (gradients)
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # 3. Update the weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        """
        Makes predictions using the trained model.

        Args:
            X (pd.DataFrame or np.ndarray): The input features for which to make predictions.
        
        Returns:
            np.ndarray: The predicted values.
        """
        X = np.array(X)
        return np.dot(X, self.weights) + self.bias


# This block allows you to run the script directly to train the model
if __name__ == "__main__":
    print("--- Starting Model Training ---")
    
    # 1. Load the TRAINING data
    # This path is now updated to use the dedicated training set.
    try:
        training_data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'training_set.csv')
        df = pd.read_csv(training_data_path)
        print(f"Successfully loaded training data from: {training_data_path}")
    except FileNotFoundError:
        print("Training data file not found. Please run split_data.py first.")
        exit()

    # 2. Prepare the data for the model
    # Drop the non-numeric 'Country' column as it cannot be used in the model directly
    df_numeric = df.drop('Country', axis=1)

    # Separate features (X) from the target (y)
    X = df_numeric.drop('Life expectancy ', axis=1)
    y = df_numeric['Life expectancy ']

    # Feature Scaling (Important for Gradient Descent)
    # We scale features to be between 0 and 1 to help the algorithm converge faster.
    X_scaled = (X - X.min()) / (X.max() - X.min())

    # 3. Train the Linear Regression model
    model = LinearRegressionScratch(learning_rate=0.01, n_iterations=2000)
    model.fit(X_scaled, y)

    # 4. Display the learned weights
    print("\n--- Model Training Complete ---")
    print(f"Bias (Intercept): {model.bias:.4f}")
    
    # Create a DataFrame to neatly display the feature names and their learned weights
    weights_df = pd.DataFrame({
        'Feature': X.columns,
        'Weight': model.weights
    }).sort_values(by='Weight', ascending=False)
    
    print("\nLearned Feature Weights:")
    print(weights_df)

    # 5. Save the trained model to a .pkl file
    # This model can be loaded later by predict.py
    try:
        model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'regression_model1.pkl')
        os.makedirs(os.path.dirname(model_path), exist_ok=True) # Create models dir if needed
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"\nModel successfully saved to: {model_path}")
    except Exception as e:
        print(f"\nError saving the model: {e}")

