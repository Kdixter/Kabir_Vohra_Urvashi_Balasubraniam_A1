import os
import numpy as np
import pickle
from typing import Tuple
import sys

# Import the RandomForest class
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from random_forest import RandomForest


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


def save_model(model: RandomForest, model_path: str) -> None:
    """
    Save the trained model using pickle.
    
    Args:
        model: Trained RandomForest model
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
    """Main function to train the Random Forest model."""
    
    # Define paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_path = os.path.join(project_root, 'data', 'final_train_data_processed.csv')
    model_path = os.path.join(project_root, 'models', 'random_forest_baseline.pkl')
    
    print("=" * 60)
    print("LAPTOP PRICE PREDICTION - RANDOM FOREST BASELINE")
    print("=" * 60)
    
    try:
        # Load training data
        X_train, y_train = load_data(data_path)
        
        # Initialize and train the Random Forest model
        model = RandomForest(
            n_estimators=50,      # Number of trees (reduced for faster training)
            max_depth=10,         # Maximum depth of each tree
            min_samples_split=5,  # Minimum samples to split a node
            min_samples_leaf=2,   # Minimum samples in a leaf
            max_features=None,    # Use sqrt of total features (auto)
            random_state=42       # For reproducibility
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
