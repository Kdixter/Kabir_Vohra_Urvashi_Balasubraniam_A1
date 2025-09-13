import numpy as np
from typing import List, Tuple, Optional
import random


class DecisionTreeNode:
    """
    A decision tree node for regression.
    """
    
    def __init__(self, feature_idx: Optional[int] = None, threshold: Optional[float] = None,
                 left: Optional['DecisionTreeNode'] = None, right: Optional['DecisionTreeNode'] = None,
                 value: Optional[float] = None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # For leaf nodes (mean of target values)
    
    def is_leaf(self) -> bool:
        """Check if this node is a leaf node."""
        return self.value is not None


class DecisionTree:
    """
    A simplified decision tree implementation for regression.
    """
    
    def __init__(self, max_depth: int = 10, min_samples_split: int = 2, min_samples_leaf: int = 1,
                 max_features: Optional[int] = None, random_state: Optional[int] = None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.root: Optional[DecisionTreeNode] = None
        
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)
    
    def _calculate_mse(self, y: np.ndarray) -> float:
        """Calculate mean squared error."""
        if len(y) == 0:
            return 0.0
        mean_val = np.mean(y)
        return np.mean((y - mean_val) ** 2)
    
    def _find_best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[int, float, float]:
        """
        Find the best split for the given data.
        
        Returns:
            Tuple of (feature_idx, threshold, mse_reduction)
        """
        best_feature = None
        best_threshold = None
        best_mse_reduction = 0.0
        
        n_samples, n_features = X.shape
        
        # Randomly select features to consider
        if self.max_features is not None:
            feature_indices = random.sample(range(n_features), min(self.max_features, n_features))
        else:
            feature_indices = range(n_features)
        
        # Calculate current MSE
        current_mse = self._calculate_mse(y)
        
        for feature_idx in feature_indices:
            # Get unique values for this feature
            feature_values = np.unique(X[:, feature_idx])
            
            for threshold in feature_values:
                # Split data
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                
                left_y = y[left_mask]
                right_y = y[right_mask]
                
                # Check minimum samples constraint
                if len(left_y) < self.min_samples_leaf or len(right_y) < self.min_samples_leaf:
                    continue
                
                # Calculate MSE for split
                left_mse = self._calculate_mse(left_y)
                right_mse = self._calculate_mse(right_y)
                
                # Weighted MSE
                weighted_mse = (len(left_y) * left_mse + len(right_y) * right_mse) / n_samples
                
                # Calculate MSE reduction
                mse_reduction = current_mse - weighted_mse
                
                if mse_reduction > best_mse_reduction:
                    best_mse_reduction = mse_reduction
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_mse_reduction
    
    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> DecisionTreeNode:
        """Recursively build the decision tree."""
        n_samples = len(y)
        
        # Stopping conditions
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or
            n_samples < self.min_samples_leaf or
            len(np.unique(y)) == 1):
            return DecisionTreeNode(value=np.mean(y))
        
        # Find best split
        feature_idx, threshold, mse_reduction = self._find_best_split(X, y)
        
        # If no good split found, create leaf node
        if mse_reduction <= 0:
            return DecisionTreeNode(value=np.mean(y))
        
        # Split data
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
        
        # Recursively build left and right subtrees
        left_tree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_tree = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return DecisionTreeNode(
            feature_idx=feature_idx,
            threshold=threshold,
            left=left_tree,
            right=right_tree
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DecisionTree':
        """Train the decision tree."""
        self.root = self._build_tree(X, y)
        return self
    
    def _predict_single(self, x: np.ndarray, node: DecisionTreeNode) -> float:
        """Predict a single sample."""
        if node.is_leaf():
            return node.value
        
        if x[node.feature_idx] <= node.threshold:
            return self._predict_single(x, node.left)
        else:
            return self._predict_single(x, node.right)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if self.root is None:
            raise ValueError("Tree must be trained before making predictions")
        
        predictions = np.array([self._predict_single(x, self.root) for x in X])
        return predictions


class RandomForest:
    """
    A simplified Random Forest implementation for regression.
    """
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 10, min_samples_split: int = 2,
                 min_samples_leaf: int = 1, max_features: Optional[int] = None, 
                 random_state: Optional[int] = None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.trees: List[DecisionTree] = []
        self.feature_means: Optional[np.ndarray] = None
        self.feature_stds: Optional[np.ndarray] = None
        self.target_mean: Optional[float] = None
        self.target_std: Optional[float] = None
        
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)
    
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
    
    def _bootstrap_sample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create a bootstrap sample."""
        n_samples = len(y)
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices]
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForest':
        """Train the Random Forest."""
        print(f"Training Random Forest with {self.n_estimators} trees...")
        
        # Clean data
        X_clean = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        y_clean = np.nan_to_num(y, nan=np.mean(y), posinf=np.max(y), neginf=np.min(y))
        
        # Normalize features and target
        X_normalized = self._normalize_features(X_clean, fit=True)
        y_normalized = self._normalize_target(y_clean, fit=True)
        
        # Set max_features if not specified (use sqrt of total features)
        if self.max_features is None:
            self.max_features = int(np.sqrt(X_normalized.shape[1]))
        
        print(f"Features per tree: {self.max_features}")
        
        # Train each tree
        for i in range(self.n_estimators):
            if i % 10 == 0:
                print(f"Training tree {i+1}/{self.n_estimators}")
            
            # Create bootstrap sample
            X_bootstrap, y_bootstrap = self._bootstrap_sample(X_normalized, y_normalized)
            
            # Create and train tree
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=self.random_state + i if self.random_state is not None else None
            )
            
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)
        
        print("Random Forest training completed!")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the Random Forest."""
        if not self.trees:
            raise ValueError("Random Forest must be trained before making predictions")
        
        # Clean and normalize input data
        X_clean = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        X_normalized = self._normalize_features(X_clean, fit=False)
        
        # Get predictions from all trees
        tree_predictions = np.array([tree.predict(X_normalized) for tree in self.trees])
        
        # Average predictions from all trees
        predictions_normalized = np.mean(tree_predictions, axis=0)
        
        # Convert back to original scale
        return self._denormalize_target(predictions_normalized)
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance (simplified version)."""
        # This is a simplified implementation
        # In a full implementation, you'd track feature usage across all trees
        n_features = len(self.feature_means)
        importance = np.zeros(n_features)
        
        # For now, return uniform importance
        # A full implementation would count how often each feature is used for splitting
        importance[:] = 1.0 / n_features
        
        return importance
