import os
import numpy as np
import pickle
from typing import Tuple
import sys

# Import the LinearRegression class with L2 regularization support
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_model import LinearRegression


def load_data(data_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and prepare the training data.
    
    Args:
        data_path: Path to the CSV file
        
    Returns:
        Tuple of (features, target)
    """
    print(f"Loading data from: {data_path}")
    
    # Load data using numpy
    data = np.genfromtxt(data_path, delimiter=',', skip_header=1)
    
    # First column is the target (Price), rest are features
    target = data[:, 0]
    features = data[:, 1:]
    
    print(f"Data shape: {data.shape}")
    print(f"Features shape: {features.shape}")
    print(f"Target shape: {target.shape}")
    print(f"Target statistics - Min: {target.min():.2f}, Max: {target.max():.2f}, Mean: {target.mean():.2f}")
    
    return features, target


def save_model(model: LinearRegression, model_path: str) -> None:
    """
    Save the trained model using pickle.
    
    Args:
        model: Trained LinearRegression model with L2 regularization
        model_path: Path where to save the model
    """
    print(f"Saving model to: {model_path}")
    
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Save model
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print("Model saved successfully!")


def main():
    """Main function to train the Ridge Regression model."""
    
    # Define paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_path = os.path.join(project_root, 'data', 'final_train_data_processed.csv')
    model_path = os.path.join(project_root, 'models', 'ridge_regression_baseline.pkl')
    
    print("=" * 60)
    print("LAPTOP PRICE PREDICTION - RIDGE REGRESSION BASELINE")
    print("=" * 60)
    
    try:
        # Load training data
        X_train, y_train = load_data(data_path)
        
        # Initialize and train the Ridge Regression model
        model = LinearRegression(
            learning_rate=0.01,   # Standard learning rate for normalized data
            max_iterations=2000,  # More iterations for convergence
            tolerance=1e-6,
            regularization=0.1    # L2 regularization parameter (Ridge regression)
        )
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Get training metrics
        # Clean the data first (same as in training)
        X_clean = np.nan_to_num(X_train, nan=0.0, posinf=1e6, neginf=-1e6)
        y_clean = np.nan_to_num(y_train, nan=np.mean(y_train), posinf=np.max(y_train), neginf=np.min(y_train))
        
        final_predictions = model.predict(X_clean)
        mse = np.mean((y_clean - final_predictions) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_clean - final_predictions))
        
        # Calculate R-squared
        ss_res = np.sum((y_clean - final_predictions) ** 2)
        ss_tot = np.sum((y_clean - np.mean(y_clean)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        print("\n" + "=" * 60)
        print("TRAINING RESULTS")
        print("=" * 60)
        print(f"Final MSE: {mse:.2f}")
        print(f"Final RMSE: {rmse:.2f}")
        print(f"Final MAE: {mae:.2f}")
        print(f"R-squared: {r_squared:.4f}")
        
        # Save the model
        save_model(model, model_path)
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise


if __name__ == "__main__":
    main()
