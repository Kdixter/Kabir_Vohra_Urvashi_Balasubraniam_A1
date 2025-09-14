import os
import numpy as np
import pickle
from typing import Tuple, Optional
from itertools import combinations_with_replacement


class PolynomialRegression:
    def __init__(self, degree: int = 2, learning_rate: float = 0.001, max_iterations: int = 1000, 
                 tolerance: float = 1e-6, regularization: float = 0.0):
        """
        Initialize the Polynomial Regression model.
        
        Args:
            degree: Degree of polynomial features (2 in my case)
            learning_rate: Learning rate for gradient descent
            max_iterations: Maximum number of iterations (usualluy like 2000)
            tolerance: Convergence tolerance
            regularization: L2 regularization 
        """
        self.degree = degree
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.regularization = regularization
        self.weights: Optional[np.ndarray] = None
        self.bias: Optional[float] = None
        self.training_losses: list = []
        self.feature_means: Optional[np.ndarray] = None
        self.feature_stds: Optional[np.ndarray] = None
        self.target_mean: Optional[float] = None
        self.target_std: Optional[float] = None
        self.poly_feature_indices: Optional[list] = None
        self.original_n_features: Optional[int] = None
        
   # function to get the features in polynomial form     
    def _generate_polynomial_features(self, X: np.ndarray) -> np.ndarray:
        
        n_samples, n_features = X.shape
        self.original_n_features = n_features
        
        # Start with original features
        poly_features = [X]
        
        # Generate polynomial features for each degree
        for degree in range(2, self.degree + 1):
            # Generate all combinations of features with replacement up to current degree
            for indices in combinations_with_replacement(range(n_features), degree):
                # Create polynomial feature by multiplying selected features
                poly_feature = np.ones(n_samples)
                for idx in indices:
                    poly_feature *= X[:, idx]
                poly_features.append(poly_feature.reshape(-1, 1))
        
        # Combine all polynomial features
        poly_X = np.column_stack(poly_features)
        
        print(f"Original features: {n_features}")
        print(f"Polynomial features (degree {self.degree}): {poly_X.shape[1]}")
        
        return poly_X
    
    def _normalize_features(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        
        if fit:
            self.feature_means = np.mean(X, axis=0)
            self.feature_stds = np.std(X, axis=0)
            # Avoid division by zero
            self.feature_stds[self.feature_stds == 0] = 1
        
        return (X - self.feature_means) / self.feature_stds
    
    def _normalize_target(self, y: np.ndarray, fit: bool = False) -> np.ndarray:
        
        if fit:
            self.target_mean = np.mean(y)
            self.target_std = np.std(y)
            if self.target_std == 0:
                self.target_std = 1
        
        return (y - self.target_mean) / self.target_std
    
    def _denormalize_target(self, y_normalized: np.ndarray) -> np.ndarray:
        
        return y_normalized * self.target_std + self.target_mean
    
    def _compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        
        predictions = X @ self.weights + self.bias
        mse = np.mean((y - predictions) ** 2)
        
        # Add L2 regularization
        if self.regularization > 0:
            l2_penalty = self.regularization * np.sum(self.weights ** 2)
            mse += l2_penalty
        
        return mse
    
    # pretty much same as in linear regression, the formula is slightly different
    def _compute_gradients(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        
        n_samples = X.shape[0]
        predictions = X @ self.weights + self.bias
        
        # Compute error
        error = predictions - y
        
        # Compute gradients
        weight_gradients = (2 / n_samples) * X.T @ error
        
        # Add L2 regularization gradient
        if self.regularization > 0:
            weight_gradients += 2 * self.regularization * self.weights
        
        bias_gradient = (2 / n_samples) * np.sum(error)
        
        return weight_gradients, bias_gradient
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'PolynomialRegression':
        # check for NaN or infinite values in input data
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            print("Warning: NaN or infinite values found in features")
            X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            print("Warning: NaN or infinite values found in target")
            y = np.nan_to_num(y, nan=np.mean(y), posinf=np.max(y), neginf=np.min(y))
        
        # Generate polynomial features
        print(f"Generating polynomial features of degree {self.degree}...")
        X_poly = self._generate_polynomial_features(X)
        
        # Normalize polynomial features and target
        X_normalized = self._normalize_features(X_poly, fit=True)
        y_normalized = self._normalize_target(y, fit=True)
        
        # Check for NaN or infinite values after normalization, very important
        if np.any(np.isnan(X_normalized)) or np.any(np.isinf(X_normalized)):
            print("Warning: NaN or infinite values found in normalized features")
            X_normalized = np.nan_to_num(X_normalized, nan=0.0, posinf=1.0, neginf=-1.0)
        
        if np.any(np.isnan(y_normalized)) or np.any(np.isinf(y_normalized)):
            print("Warning: NaN or infinite values found in normalized target")
            y_normalized = np.nan_to_num(y_normalized, nan=0.0, posinf=1.0, neginf=-1.0)
        
        n_samples, n_features = X_normalized.shape
        
        # Initialize weights and bias with smaller values
        self.weights = np.random.normal(0, 0.001, n_features)
        self.bias = 0.0
        
        print(f"Training Polynomial Regression model...")
        print(f"Training samples: {n_samples}, Polynomial features: {n_features}")
        print(f"Learning rate: {self.learning_rate}, Max iterations: {self.max_iterations}")
        print(f"Regularization: {self.regularization}")
        
        # Gradient descent
        for iteration in range(self.max_iterations):
            # Compute predictions on normalized data
            predictions_normalized = X_normalized @ self.weights + self.bias
            
            # Check for NaN in predictions
            if np.any(np.isnan(predictions_normalized)) or np.any(np.isinf(predictions_normalized)):
                print(f"Warning: NaN or infinite values in predictions at iteration {iteration}")
                predictions_normalized = np.nan_to_num(predictions_normalized, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Compute loss on normalized data
            loss = self._compute_loss(X_normalized, y_normalized)
            self.training_losses.append(loss)
            
            # Check for NaN in loss (can skip)
            if np.isnan(loss) or np.isinf(loss):
                print(f"Warning: NaN or infinite loss at iteration {iteration}")
                print(f"Weights range: [{self.weights.min():.6f}, {self.weights.max():.6f}]")
                print(f"Bias: {self.bias:.6f}")
                break
            
            weight_grads, bias_grad = self._compute_gradients(X_normalized, y_normalized)
            
            if np.any(np.isnan(weight_grads)) or np.isnan(bias_grad) or np.any(np.isinf(weight_grads)) or np.isinf(bias_grad):
                print(f"Warning: NaN or infinite gradients at iteration {iteration}")
                break
            
            # updation of paramters in each iteration
            self.bias -= self.learning_rate * bias_grad
            self.weights -= self.learning_rate * weight_grads
            
            #Convergence Check:
            if iteration > 0 and abs(self.training_losses[-2] - self.training_losses[-1]) < self.tolerance:
                print(f"Converged after {iteration + 1} iterations")
                break
            
            # Print progress:
            if iteration % 100 == 0:
                print(f"Iteration {iteration}: Loss = {loss:.6f}")
        
        print(f"Training completed. Final loss: {self.training_losses[-1]:.6f}")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None or self.bias is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Clean input data
        X_clean = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Generate polynomial features
        X_poly = self._generate_polynomial_features(X_clean)
        
        # Normalize features using stored statistics
        X_normalized = self._normalize_features(X_poly, fit=False)
        
        # Make predictions on normalized features
        predictions_normalized = X_normalized @ self.weights + self.bias
        
        # Convert back to original scale
        return self._denormalize_target(predictions_normalized)
    
    def get_coefficients(self) -> dict:
        return {
            'weights': self.weights.copy() if self.weights is not None else None,
            'bias': self.bias
        }
    
    def get_training_losses(self) -> list:
        """Get training loss history."""
        return self.training_losses.copy()


def load_data(data_path: str) -> Tuple[np.ndarray, np.ndarray]:
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


def save_model(model: PolynomialRegression, model_path: str) -> None:
    print(f"Saving model to: {model_path}")
    
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Save model
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print("Model saved successfully!")


def main():
    """Main function to train the polynomial regression model."""
    
    # Define paths (paths may change for different systems)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_path = os.path.join(project_root, 'data', 'final_train_data_processed.csv')
    model_path = os.path.join(project_root, 'models', 'polynomial_regression_baseline.pkl')
    
    print("=" * 60)
    print("LAPTOP PRICE PREDICTION - POLYNOMIAL REGRESSION BASELINE")
    print("=" * 60)
    
    try:
        # Load training data
        X_train, y_train = load_data(data_path)
        
        # Initialize and train the model
        model = PolynomialRegression(
            degree=2,              # Quadratic polynomial features (tried for others, this works best)
            learning_rate=0.0001,  # Lower learning rate for polynomial features
            max_iterations=3000,   # More iterations for convergence for polynomial
            tolerance=1e-6,
            regularization=0.01    # L2 regularization
        )
        
        # Train the model
        model.fit(X_train, y_train)
        
        X_clean = np.nan_to_num(X_train, nan=0.0, posinf=1e6, neginf=-1e6)
        y_clean = np.nan_to_num(y_train, nan=np.mean(y_train), posinf=np.max(y_train), neginf=np.min(y_train))
        
        final_predictions = model.predict(X_clean)
        mse = np.mean((y_clean - final_predictions) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_clean - final_predictions))
        
        # Calculate R^2
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
