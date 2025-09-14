import os
import numpy as np
import pickle
from typing import Tuple, Optional
import pandas as pd


class LinearRegressionEngineered:
    
    def __init__(self, learning_rate: float = 0.01, max_iterations: int = 2000, tolerance: float = 1e-6, 
                 regularization: float = 0.0, degree: int = 2):
        
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.regularization = regularization
        self.degree = degree
        self.weights: Optional[np.ndarray] = None
        self.bias: Optional[float] = None
        self.training_losses: list = []
        self.feature_means: Optional[np.ndarray] = None
        self.feature_stds: Optional[np.ndarray] = None
        self.target_mean: Optional[float] = None
        self.target_std: Optional[float] = None
        self.original_feature_names: Optional[list] = None
        self.engineered_feature_names: Optional[list] = None
        
    def _add_bias_term(self, X: np.ndarray) -> np.ndarray:
        """Add bias term (column of ones) to the feature matrix."""
        return np.column_stack([np.ones(X.shape[0]), X])
    
    def _create_interaction_features(self, X: np.ndarray, feature_names: list) -> Tuple[np.ndarray, list]:
        
        enhanced_features = [X]
        enhanced_names = list(feature_names)
        
        # Find indices of important features
        feature_dict = {name: idx for idx, name in enumerate(feature_names)}
        
        # Key interaction features based on domain knowledge
        interactions = [
            ('GDP', 'Schooling'),  # Wealth-education interaction
            ('BMI ', 'Adult Mortality'),  # Health interaction
            ('Alcohol', 'GDP'),  # Wealth-health behavior
            ('Polio', 'Diphtheria '),  # Vaccination coverage
            ('Year', 'Status'),  # Development over time
            ('Income composition of resources', 'Schooling'),  # Development indicators
            ('Total expenditure', 'GDP'),  # Healthcare spending vs wealth
        ]
        
        for feat1, feat2 in interactions:
            if feat1 in feature_dict and feat2 in feature_dict:
                idx1, idx2 = feature_dict[feat1], feature_dict[feat2]
                interaction = X[:, idx1] * X[:, idx2]
                enhanced_features.append(interaction.reshape(-1, 1))
                enhanced_names.append(f"{feat1}_x_{feat2}")
                print(f"Created interaction: {feat1}_x_{feat2}")
        
        return np.column_stack(enhanced_features), enhanced_names
    
    def _create_polynomial_features(self, X: np.ndarray, feature_names: list) -> Tuple[np.ndarray, list]:
        
        polynomial_features = [X]
        polynomial_names = list(feature_names)
        
        # Find indices of key features for polynomial expansion
        feature_dict = {name: idx for idx, name in enumerate(feature_names)}
        
        # Key features for polynomial expansion
        key_features = ['GDP', 'Schooling', 'BMI ', 'Income composition of resources', 'Adult Mortality']
        
        for feat_name in key_features:
            if feat_name in feature_dict:
                idx = feature_dict[feat_name]
                # Create polynomial features up to degree
                for degree in range(2, self.degree + 1):
                    poly_feat = X[:, idx] ** degree
                    polynomial_features.append(poly_feat.reshape(-1, 1))
                    polynomial_names.append(f"{feat_name}^{degree}")
                    print(f"Created polynomial: {feat_name}^{degree}")
        
        return np.column_stack(polynomial_features), polynomial_names
    
    def _normalize_features(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Normalize features using z-score normalization."""
        if fit:
            self.feature_means = np.mean(X, axis=0)
            self.feature_stds = np.std(X, axis=0)
            # Avoid division by zero
            self.feature_stds[self.feature_stds == 0] = 1
        
        return (X - self.feature_means) / self.feature_stds
    
    def _normalize_target(self, y: np.ndarray, fit: bool = False) -> np.ndarray:
        """Normalize target using z-score normalization."""
        if fit:
            self.target_mean = np.mean(y)
            self.target_std = np.std(y)
            if self.target_std == 0:
                self.target_std = 1
        
        return (y - self.target_mean) / self.target_std
    
    def _denormalize_target(self, y_normalized: np.ndarray) -> np.ndarray:
        """Convert normalized predictions back to original scale."""
        return y_normalized * self.target_std + self.target_mean
    
    def _compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute mean squared error loss with optional regularization."""
        predictions = self.predict(X)
        mse = np.mean((y - predictions) ** 2)
        
        # Add regularization penalty if specified
        if self.regularization > 0 and self.weights is not None:
            l2_penalty = self.regularization * np.sum(self.weights ** 2)
            mse += l2_penalty
        
        return mse
    
    def _compute_gradients(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        
        n_samples = X.shape[0]
        # Use X without bias term for gradient computation
        X_features = X[:, 1:]  # Remove the bias column
        predictions = X @ np.concatenate([[self.bias], self.weights])
        
        # Compute error
        error = predictions - y
        
        # Compute gradients
        weight_gradients = (2 / n_samples) * X_features.T @ error
        
        # Add L2 regularization gradient if specified
        if self.regularization > 0:
            weight_gradients += 2 * self.regularization * self.weights
        
        bias_gradient = (2 / n_samples) * np.sum(error)
        
        return weight_gradients, bias_gradient
    
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: list = None) -> 'LinearRegressionEngineered':
       
        print(f"Starting linear regression training with engineered features (degree {self.degree})...")
        
        # Store original feature names
        self.original_feature_names = feature_names if feature_names else [f"feature_{i}" for i in range(X.shape[1])]
        
        # Check for NaN or infinite values in input data
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            print("Warning: NaN or infinite values found in features")
            X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            print("Warning: NaN or infinite values found in target")
            y = np.nan_to_num(y, nan=np.mean(y), posinf=np.max(y), neginf=np.min(y))
        
        # Create interaction features
        print("Creating interaction features...")
        X_interactions, interaction_names = self._create_interaction_features(X, self.original_feature_names)
        
        # Create polynomial features
        print("Creating polynomial features...")
        X_engineered, engineered_names = self._create_polynomial_features(X_interactions, interaction_names)
        
        # Store engineered feature names
        self.engineered_feature_names = engineered_names
        
        print(f"Original features: {X.shape[1]}")
        print(f"After interactions: {X_interactions.shape[1]}")
        print(f"After polynomials: {X_engineered.shape[1]}")
        
        # Normalize features and target
        X_normalized = self._normalize_features(X_engineered, fit=True)
        y_normalized = self._normalize_target(y, fit=True)
        
        # Check for NaN or infinite values after normalization
        if np.any(np.isnan(X_normalized)) or np.any(np.isinf(X_normalized)):
            print("Warning: NaN or infinite values found in normalized features")
            X_normalized = np.nan_to_num(X_normalized, nan=0.0, posinf=1.0, neginf=-1.0)
        
        if np.any(np.isnan(y_normalized)) or np.any(np.isinf(y_normalized)):
            print("Warning: NaN or infinite values found in normalized target")
            y_normalized = np.nan_to_num(y_normalized, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Add bias term to features
        X_with_bias = self._add_bias_term(X_normalized)
        n_samples, n_features = X_with_bias.shape
        
        # Initialize weights and bias with smaller values
        self.weights = np.random.normal(0, 0.001, n_features - 1)  # Exclude bias from weights
        self.bias = 0.0
        
        print(f"Training Linear Regression with Engineered Features...")
        print(f"Training samples: {n_samples}, Features: {n_features - 1}")
        print(f"Learning rate: {self.learning_rate}, Max iterations: {self.max_iterations}")
        print(f"Regularization: {self.regularization}")
        print(f"Polynomial degree: {self.degree}")
        
        # Gradient descent
        for iteration in range(self.max_iterations):
            # Compute predictions on normalized data
            predictions_normalized = X_with_bias @ np.concatenate([[self.bias], self.weights])
            
            # Check for NaN in predictions
            if np.any(np.isnan(predictions_normalized)) or np.any(np.isinf(predictions_normalized)):
                print(f"Warning: NaN or infinite values in predictions at iteration {iteration}")
                predictions_normalized = np.nan_to_num(predictions_normalized, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Compute loss on normalized data
            loss = np.mean((y_normalized - predictions_normalized) ** 2)
            self.training_losses.append(loss)
            
            # Check for NaN in loss
            if np.isnan(loss) or np.isinf(loss):
                print(f"Warning: NaN or infinite loss at iteration {iteration}")
                print(f"Weights range: [{self.weights.min():.6f}, {self.weights.max():.6f}]")
                print(f"Bias: {self.bias:.6f}")
                break
            
            # Compute gradients
            weight_grads, bias_grad = self._compute_gradients(X_with_bias, y_normalized)
            
            # Check for NaN in gradients
            if np.any(np.isnan(weight_grads)) or np.isnan(bias_grad) or np.any(np.isinf(weight_grads)) or np.isinf(bias_grad):
                print(f"Warning: NaN or infinite gradients at iteration {iteration}")
                break
            
            # Update parameters
            self.bias -= self.learning_rate * bias_grad
            self.weights -= self.learning_rate * weight_grads
            
            # Check for convergence
            if iteration > 0 and abs(self.training_losses[-2] - self.training_losses[-1]) < self.tolerance:
                print(f"Converged after {iteration + 1} iterations")
                break
            
            # Print progress every 100 iterations
            if iteration % 100 == 0:
                print(f"Iteration {iteration}: Loss = {loss:.6f}")
        
        print(f"Training completed. Final loss: {self.training_losses[-1]:.6f}")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        
        if self.weights is None or self.bias is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Create the same feature engineering as in training
        # Create interaction features
        X_interactions, _ = self._create_interaction_features(X, self.original_feature_names)
        
        # Create polynomial features
        X_engineered, _ = self._create_polynomial_features(X_interactions, self.original_feature_names + 
                                                         [name for name in self.engineered_feature_names if '_x_' in name])
        
        # Normalize features using stored statistics
        X_normalized = self._normalize_features(X_engineered, fit=False)
        
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
        return self.training_losses.copy()
    
    def get_feature_names(self) -> list:
        return self.engineered_feature_names


def load_data(data_path: str) -> Tuple[np.ndarray, np.ndarray, list]:
    
    print(f"Loading data from: {data_path}")
    
    # Load data using pandas to handle mixed data types properly
    df = pd.read_csv(data_path)
    
    # Get feature names (excluding target)
    feature_names = df.columns[:-1].tolist()
    
    # Convert boolean columns (country one-hot encoded features) to integers
    for col in df.columns:
        if df[col].dtype == 'object':
            # Convert 'True'/'False' strings to 1/0
            df[col] = df[col].map({'True': 1, 'False': 0})
    
    # Convert to numpy arrays
    data = df.values.astype(float)
    
    # Last column is the target (Life expectancy), rest are features
    target = data[:, -1]
    features = data[:, :-1]
    
    print(f"Data shape: {data.shape}")
    print(f"Features shape: {features.shape}")
    print(f"Target shape: {target.shape}")
    print(f"Target statistics - Min: {target.min():.2f}, Max: {target.max():.2f}, Mean: {target.mean():.2f}")
    
    return features, target, feature_names


def save_model(model: LinearRegressionEngineered, model_path: str) -> None:
    print(f"Saving model to: {model_path}")
    
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Save model
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print("Model saved successfully!")


def main():
    
    # Define paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_path = os.path.join(project_root, 'data', 'life_expectancy_train_processed.csv')
    model_path = os.path.join(project_root, 'models', 'life_expectancy_linear_engineered_model.pkl')
    
    print("=" * 60)
    print("LIFE EXPECTANCY PREDICTION - LINEAR REGRESSION WITH ENGINEERED FEATURES")
    print("=" * 60)
    
    try:
        # Load training data
        X_train, y_train, feature_names = load_data(data_path)
        
        # Initialize and train the model with engineered features
        model = LinearRegressionEngineered(
            learning_rate=0.01,   # Standard learning rate
            max_iterations=2000,  # More iterations for convergence
            tolerance=1e-6,
            regularization=0.0,   # No regularization for baseline comparison
            degree=2              # Quadratic polynomial features
        )
        
        # Train the model
        model.fit(X_train, y_train, feature_names)
        
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
        print(f"Total engineered features used: {len(model.get_feature_names())}")
        
        # Save the model
        save_model(model, model_path)
        
        print("\n" + "=" * 60)
        print("LINEAR REGRESSION WITH ENGINEERED FEATURES TRAINING COMPLETED!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise


if __name__ == "__main__":
    main()
