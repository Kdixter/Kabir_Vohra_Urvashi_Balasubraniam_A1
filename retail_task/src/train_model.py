import os
import numpy as np
import pickle
from typing import Tuple, Optional


class LinearRegression:
    def __init__(self, learning_rate: float = 0.01, max_iterations: int = 1000, tolerance: float = 1e-6,
                 regularization: float = 0.0, regularization_type: str = 'l2'):
        """
        Arguments:
            learning_rate: Learning rate for gradient descent
            max_iterations: self explanatory
            tolerance: Convergence tolerance, when the model stops iterations and finially stops, the point I have now reached
            regularization: Regularization parameter (0.0 = no regularization)
            regularization_type: Type of regularization ('l1', 'l2') or Elastic net, although I am not that confident in it, so I have not implemented
        """

        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.regularization = regularization
        self.regularization_type = regularization_type.lower()
        self.weights: Optional[np.ndarray] = None
        self.bias: Optional[float] = None
        self.training_losses: list = []
        self.feature_means: Optional[np.ndarray] = None
        self.feature_stds: Optional[np.ndarray] = None
        self.target_mean: Optional[float] = None
        self.target_std: Optional[float] = None
        
    def _add_bias_term(self, X: np.ndarray) -> np.ndarray:
        return np.column_stack([np.ones(X.shape[0]), X])
    
    def _normalize_features(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        if fit:
            self.feature_means = np.mean(X, axis=0)
            self.feature_stds = np.std(X, axis=0)
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
        predictions = self.predict(X)
        mse = np.mean((y - predictions) ** 2)
        
        if self.regularization > 0 and self.weights is not None:
            if self.regularization_type == 'l1':
                l1_penalty = self.regularization * np.sum(np.abs(self.weights))
                mse += l1_penalty
            elif self.regularization_type == 'l2':
                l2_penalty = self.regularization * np.sum(self.weights ** 2)
                mse += l2_penalty
            elif self.regularization_type == 'elastic_net':
                l1_penalty = self.regularization * np.sum(np.abs(self.weights))
                l2_penalty = self.regularization * np.sum(self.weights ** 2)
                mse += l1_penalty + l2_penalty
        
        return mse
    
    def _compute_gradients(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        n_samples = X.shape[0]
        X_features = X[:, 1:]
        predictions = X @ np.concatenate([[self.bias], self.weights])
        error = predictions - y
        weight_gradients = (2 / n_samples) * X_features.T @ error
        
        if self.regularization > 0:
            if self.regularization_type == 'l1':
                epsilon = 1e-8
                weight_gradients += self.regularization * np.sign(self.weights) * (np.abs(self.weights) > epsilon)
            
            elif self.regularization_type == 'l2':
                weight_gradients += 2 * self.regularization * self.weights
            
            elif self.regularization_type == 'elastic_net':
                epsilon = 1e-8
                l1_grad = self.regularization * np.sign(self.weights) * (np.abs(self.weights) > epsilon)
                l2_grad = 2 * self.regularization * self.weights
                weight_gradients += l1_grad + l2_grad
        
        bias_gradient = (2 / n_samples) * np.sum(error)
        
        return weight_gradients, bias_gradient
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegression':
        
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            print("Warning: NaN or infinite values found in features")
            X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            print("Warning: NaN or infinite values found in target")
            y = np.nan_to_num(y, nan=np.mean(y), posinf=np.max(y), neginf=np.min(y))
        
        X_normalized = self._normalize_features(X, fit=True)
        y_normalized = self._normalize_target(y, fit=True)
        
        if np.any(np.isnan(X_normalized)) or np.any(np.isinf(X_normalized)):
            print("Warning: NaN or infinite values found in normalized features")
            X_normalized = np.nan_to_num(X_normalized, nan=0.0, posinf=1.0, neginf=-1.0)
        
        if np.any(np.isnan(y_normalized)) or np.any(np.isinf(y_normalized)):
            print("Warning: NaN or infinite values found in normalized target")
            y_normalized = np.nan_to_num(y_normalized, nan=0.0, posinf=1.0, neginf=-1.0)
        
        X_with_bias = self._add_bias_term(X_normalized)
        n_samples, n_features = X_with_bias.shape
        
        self.weights = np.random.normal(0, 0.001, n_features - 1)
        self.bias = 0.0
        
        print(f"Training Linear Regression model...")
        print(f"Training samples: {n_samples}, Features: {n_features - 1}")
        print(f"Learning rate: {self.learning_rate}, Max iterations: {self.max_iterations}")
        print(f"Regularization ({self.regularization_type.upper()}): {self.regularization}")
        
        for iteration in range(self.max_iterations):
            predictions_normalized = X_with_bias @ np.concatenate([[self.bias], self.weights])
            
            if np.any(np.isnan(predictions_normalized)) or np.any(np.isinf(predictions_normalized)):
                print(f"Warning: NaN or infinite values in predictions at iteration {iteration}")
                predictions_normalized = np.nan_to_num(predictions_normalized, nan=0.0, posinf=1.0, neginf=-1.0)
            
            loss = np.mean((y_normalized - predictions_normalized) ** 2)
            self.training_losses.append(loss)
            
            if np.isnan(loss) or np.isinf(loss):
                print(f"Warning: NaN or infinite loss at iteration {iteration}")
                print(f"Weights range: [{self.weights.min():.6f}, {self.weights.max():.6f}]")
                print(f"Bias: {self.bias:.6f}")
                break
            
            weight_grads, bias_grad = self._compute_gradients(X_with_bias, y_normalized)
            
            if np.any(np.isnan(weight_grads)) or np.isnan(bias_grad) or np.any(np.isinf(weight_grads)) or np.isinf(bias_grad):
                print(f"Warning: NaN or infinite gradients at iteration {iteration}")
                break
            
            self.bias -= self.learning_rate * bias_grad
            self.weights -= self.learning_rate * weight_grads
            
            if iteration > 0 and abs(self.training_losses[-2] - self.training_losses[-1]) < self.tolerance:
                print(f"Converged after {iteration + 1} iterations")
                break
            

            if iteration % 100 == 0:
                print(f"Iteration {iteration}: Loss = {loss:.6f}")
        
        print(f"Training completed. Final loss: {self.training_losses[-1]:.6f}")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None or self.bias is None:
            raise ValueError("Model must be trained before making predictions")
        
        X_normalized = self._normalize_features(X, fit=False)
        predictions_normalized = X_normalized @ self.weights + self.bias
        
        return self._denormalize_target(predictions_normalized)
    
    def get_coefficients(self) -> dict:
        return {
            'weights': self.weights.copy() if self.weights is not None else None,
            'bias': self.bias
        }
    
    def get_training_losses(self) -> list:
        return self.training_losses.copy()


class QuadraticLinearModel(LinearRegression):
    def _add_quadratic_features(self, X: np.ndarray) -> np.ndarray:
        return np.hstack((X, X**2))

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegression':
        X_transformed = self._add_quadratic_features(X)
        return super().fit(X_transformed, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_transformed = self._add_quadratic_features(X)
        return super().predict(X_transformed)


def load_data(data_path: str) -> Tuple[np.ndarray, np.ndarray]:
    
    print(f"Loading data from: {data_path}")
    
    data = np.genfromtxt(data_path, delimiter=',', skip_header=1)
    
    target = data[:, 0]
    features = data[:, 1:]
    
    print(f"Data shape: {data.shape}")
    print(f"Features shape: {features.shape}")
    print(f"Target shape: {target.shape}")
    print(f"Target statistics - Min: {target.min():.2f}, Max: {target.max():.2f}, Mean: {target.mean():.2f}")
    
    return features, target

def save_model(model: LinearRegression, model_path: str) -> None:

    print(f"Saving model to: {model_path}")
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print("Model saved successfully!")


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_path = os.path.join(project_root, 'notebooks', 'selected_features.csv')
    model_path = os.path.join(project_root, 'models', 'quadratic_linear_model.pkl')
    
    print("=" * 60)
    print("LAPTOP PRICE PREDICTION - QUADRATIC LINEAR MODEL")
    print("=" * 60)
    
    try:
        X_train, y_train = load_data(data_path)
        
        model = QuadraticLinearModel( 
            learning_rate=0.01,
            max_iterations=2000,
            tolerance=1e-6
        )
        
        model.fit(X_train, y_train)
        
        X_clean = np.nan_to_num(X_train, nan=0.0, posinf=1e6, neginf=-1e6)
        y_clean = np.nan_to_num(y_train, nan=np.mean(y_train), posinf=np.max(y_train), neginf=np.min(y_train))
        
        final_predictions = model.predict(X_clean)
        mse = np.mean((y_clean - final_predictions) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_clean - final_predictions))
        
        ss_res = np.sum((y_clean - final_predictions) ** 2)
        ss_tot = np.sum((y_clean - np.mean(y_clean)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        print("TRAINING RESULTS")
        print(f"Final MSE: {mse:.2f}")
        print(f"Final RMSE: {rmse:.2f}")
        print(f"Final MAE: {mae:.2f}")
        print(f"R-squared: {r_squared:.4f}")
        
        save_model(model, model_path)
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise


if __name__ == "__main__":
    main()