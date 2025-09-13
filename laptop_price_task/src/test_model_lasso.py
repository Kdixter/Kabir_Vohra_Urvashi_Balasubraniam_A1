import os
import numpy as np
import pickle
from typing import Tuple
import sys

# Add the current directory to Python path to import the LinearRegression class
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_model import LinearRegression


def load_test_data(data_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and prepare the test data.
    
    Args:
        data_path: Path to the CSV file
        
    Returns:
        Tuple of (features, target)
    """
    print(f"Loading test data from: {data_path}")
    
    # Load data using numpy
    data = np.genfromtxt(data_path, delimiter=',', skip_header=1)
    
    # First column is the target (Price), rest are features
    target = data[:, 0]
    features = data[:, 1:]
    
    print(f"Test data shape: {data.shape}")
    print(f"Features shape: {features.shape}")
    print(f"Target shape: {target.shape}")
    print(f"Target statistics - Min: {target.min():.2f}, Max: {target.max():.2f}, Mean: {target.mean():.2f}")
    
    return features, target


def load_model(model_path: str):
    """
    Load the trained model from pickle file.
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        Loaded LinearRegression model with L1 regularization
    """
    print(f"Loading model from: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    print("Model loaded successfully!")
    return model


def clean_data(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Clean the data by handling NaN and infinite values.
    
    Args:
        X: Feature matrix
        y: Target values
        
    Returns:
        Tuple of (cleaned_features, cleaned_target)
    """
    # Check for NaN or infinite values in input data
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        print("Warning: NaN or infinite values found in features")
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        print("Warning: NaN or infinite values found in target")
        y = np.nan_to_num(y, nan=np.mean(y), posinf=np.max(y), neginf=np.min(y))
    
    return X, y


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculate various evaluation metrics.
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        
    Returns:
        Dictionary containing evaluation metrics
    """
    # Mean Squared Error
    mse = np.mean((y_true - y_pred) ** 2)
    
    # Root Mean Squared Error
    rmse = np.sqrt(mse)
    
    # Mean Absolute Error
    mae = np.mean(np.abs(y_true - y_pred))
    
    # R-squared
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R_squared': r_squared,
        'MAPE': mape
    }


def print_sample_predictions(y_true: np.ndarray, y_pred: np.ndarray, n_samples: int = 10):
    """
    Print sample predictions for inspection.
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        n_samples: Number of samples to display
    """
    print(f"\n{'Sample Predictions':<50}")
    print("=" * 50)
    print(f"{'Index':<8} {'True Price':<15} {'Predicted Price':<18} {'Error':<15}")
    print("-" * 50)
    
    for i in range(min(n_samples, len(y_true))):
        true_val = y_true[i]
        pred_val = y_pred[i]
        error = abs(true_val - pred_val)
        print(f"{i:<8} {true_val:<15.2f} {pred_val:<18.2f} {error:<15.2f}")


def analyze_lasso_features(model: LinearRegression) -> None:
    """
    Analyze which features were selected by Lasso.
    
    Args:
        model: Trained Lasso model
    """
    if model.weights is None:
        print("Model not trained yet!")
        return
    
    # Count non-zero weights
    non_zero_mask = np.abs(model.weights) > 1e-6
    selected_features = np.sum(non_zero_mask)
    total_features = len(model.weights)
    
    print(f"\n{'Lasso Feature Selection Results':<50}")
    print("=" * 50)
    print(f"Total features: {total_features}")
    print(f"Selected features: {selected_features}")
    print(f"Removed features: {total_features - selected_features}")
    print(f"Feature reduction: {((total_features - selected_features) / total_features) * 100:.1f}%")
    
    # Show top selected features
    weights_abs = np.abs(model.weights)
    top_features = np.argsort(weights_abs)[::-1][:10]
    
    print(f"\n{'Top 10 Selected Features':<50}")
    print("-" * 50)
    print(f"{'Rank':<6} {'Feature':<15} {'Weight':<15} {'Importance':<10}")
    print("-" * 50)
    
    for i, idx in enumerate(top_features):
        if weights_abs[idx] > 1e-6:  # Only show non-zero weights
            print(f"{i+1:<6} Feature_{idx:<12} {model.weights[idx]:<15.6f} {weights_abs[idx]:<10.6f}")


def save_results_to_file(metrics: dict, y_true: np.ndarray, y_pred: np.ndarray, 
                        model: LinearRegression, project_root: str):
    """
    Save test results to a text file.
    
    Args:
        metrics: Dictionary containing evaluation metrics
        y_true: True target values
        y_pred: Predicted target values
        model: Trained Lasso model
        project_root: Path to project root directory
    """
    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    results_file = os.path.join(results_dir, 'lasso_model_test_results.txt')
    
    # Count selected features
    if model.weights is not None:
        non_zero_mask = np.abs(model.weights) > 1e-6
        selected_features = np.sum(non_zero_mask)
        total_features = len(model.weights)
    else:
        selected_features = 0
        total_features = 0
    
    with open(results_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("LAPTOP PRICE PREDICTION - LASSO REGRESSION MODEL TEST RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("MODEL INFORMATION:\n")
        f.write("-" * 40 + "\n")
        f.write("Model Type: Lasso Regression (Linear Regression with L1 Regularization)\n")
        f.write("Optimization: Gradient Descent\n")
        f.write("Regularization: L1 (Lasso)\n")
        f.write(f"Total Features: {total_features}\n")
        f.write(f"Selected Features: {selected_features}\n")
        f.write(f"Feature Reduction: {((total_features - selected_features) / total_features) * 100:.1f}%\n")
        f.write(f"Test Samples: {len(y_true)}\n\n")
        
        f.write("EVALUATION METRICS:\n")
        f.write("-" * 40 + "\n")
        f.write("Mean Squared Error (MSE): {:.2f}\n".format(metrics['MSE']))
        f.write("Root Mean Squared Error (RMSE): {:.2f}\n".format(metrics['RMSE']))
        f.write("Mean Absolute Error (MAE): {:.2f}\n".format(metrics['MAE']))
        f.write("R-squared: {:.4f}\n".format(metrics['R_squared']))
        f.write("Mean Absolute Percentage Error (MAPE): {:.2f}%\n\n".format(metrics['MAPE']))
        
        f.write("ADDITIONAL STATISTICS:\n")
        f.write("-" * 40 + "\n")
        f.write("Test samples: {}\n".format(len(y_true)))
        f.write("Prediction range: [{:.2f}, {:.2f}]\n".format(y_pred.min(), y_pred.max()))
        f.write("True value range: [{:.2f}, {:.2f}]\n".format(y_true.min(), y_true.max()))
        
        residuals = y_true - y_pred
        f.write("Residual mean: {:.2f}\n".format(residuals.mean()))
        f.write("Residual std: {:.2f}\n\n".format(residuals.std()))
        
        f.write("SAMPLE PREDICTIONS:\n")
        f.write("-" * 40 + "\n")
        f.write("{:<8} {:<15} {:<18} {:<15}\n".format("Index", "True Price", "Predicted Price", "Error"))
        f.write("-" * 56 + "\n")
        
        for i in range(min(20, len(y_true))):
            true_val = y_true[i]
            pred_val = y_pred[i]
            error = abs(true_val - pred_val)
            f.write("{:<8} {:<15.2f} {:<18.2f} {:<15.2f}\n".format(i, true_val, pred_val, error))
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("RESULTS SAVED: {}\n".format(results_file))
        f.write("=" * 80 + "\n")
    
    print(f"\nResults saved to: {results_file}")


def main():
    """Main function to test the Lasso Regression model."""
    
    # Define paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    test_data_path = os.path.join(project_root, 'data', 'final_test_data.csv')
    model_path = os.path.join(project_root, 'models', 'lasso_regression_baseline.pkl')
    
    print("=" * 60)
    print("LAPTOP PRICE PREDICTION - LASSO MODEL TESTING")
    print("=" * 60)
    
    try:
        # Load test data
        X_test, y_test = load_test_data(test_data_path)
        
        # Load trained model
        model = load_model(model_path)
        
        # Clean the test data (same cleaning as training)
        X_test_clean, y_test_clean = clean_data(X_test, y_test)
        
        # Make predictions
        print("\nGenerating predictions on test data...")
        y_pred = model.predict(X_test_clean)
        
        # Calculate evaluation metrics
        metrics = calculate_metrics(y_test_clean, y_pred)
        
        # Display results
        print("\n" + "=" * 60)
        print("TEST RESULTS")
        print("=" * 60)
        print(f"Mean Squared Error (MSE): {metrics['MSE']:.2f}")
        print(f"Root Mean Squared Error (RMSE): {metrics['RMSE']:.2f}")
        print(f"Mean Absolute Error (MAE): {metrics['MAE']:.2f}")
        print(f"R-squared: {metrics['R_squared']:.4f}")
        print(f"Mean Absolute Percentage Error (MAPE): {metrics['MAPE']:.2f}%")
        
        # Analyze Lasso feature selection
        analyze_lasso_features(model)
        
        # Print sample predictions
        print_sample_predictions(y_test_clean, y_pred, n_samples=15)
        
        # Additional statistics
        print(f"\n{'Additional Statistics':<50}")
        print("=" * 50)
        print(f"Test samples: {len(y_test_clean)}")
        print(f"Prediction range: [{y_pred.min():.2f}, {y_pred.max():.2f}]")
        print(f"True value range: [{y_test_clean.min():.2f}, {y_test_clean.max():.2f}]")
        
        # Calculate residuals
        residuals = y_test_clean - y_pred
        print(f"Residual mean: {residuals.mean():.2f}")
        print(f"Residual std: {residuals.std():.2f}")
        
        print("\n" + "=" * 60)
        print("TESTING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        # Save results to file
        save_results_to_file(metrics, y_test_clean, y_pred, model, project_root)
        
        return metrics
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        raise


if __name__ == "__main__":
    main()
