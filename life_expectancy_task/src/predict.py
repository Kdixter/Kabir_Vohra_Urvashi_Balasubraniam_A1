import pickle
import argparse
from pathlib import Path
from itertools import combinations_with_replacement
from typing import Tuple, List

import numpy as np
import pandas as pd

# Assuming data_preprocessing.py is in the same directory (src)
from data_preprocessing import (
    encode_status,
    impute_numeric_median,
    one_hot_encode_country,
    normalize_features_to_unit_range,
    drop_low_correlation_features,
    create_interaction_features,
    load_target_correlations,
    get_paths,
)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved model file.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the data CSV file.")
    parser.add_argument("--metrics_output_path", type=str, required=True, help="Path to save evaluation metrics.")
    parser.add_argument("--predictions_output_path", type=str, required=True, help="Path to save predictions.")
    return parser.parse_args()

def build_poly_feature_map(base_features: List[str], degree: int) -> List[tuple]:
    """Build a map of polynomial features."""
    index_map = list(range(len(base_features)))
    terms = []
    for d in range(1, degree + 1):
        for combo in combinations_with_replacement(index_map, d):
            name_tuple = tuple(base_features[i] for i in combo)
            terms.append((combo, name_tuple))
    return terms

def expand_polynomial_features(df: pd.DataFrame, base_features: List[str], degree: int) -> Tuple[np.ndarray, List[str]]:
    """Expand a dataframe with polynomial features."""
    terms = build_poly_feature_map(base_features, degree)
    X_base = df[base_features].to_numpy(dtype=float)
    N = X_base.shape[0]
    X_terms = []
    term_names = []
    for combo, name_tuple in terms:
        col = np.ones(N, dtype=float)
        for idx in combo:
            col = col * X_base[:, idx]
        X_terms.append(col.reshape(N, 1))
        term_names.append("*".join(name_tuple))
    X_poly = np.concatenate(X_terms, axis=1) if X_terms else np.empty((N, 0))
    return X_poly, term_names

def df_to_design_matrix_poly(df: pd.DataFrame, base_features: List[str], degree: int, target_col: str) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a dataframe to a design matrix with polynomial features."""
    X_poly, _ = expand_polynomial_features(df, base_features, degree)
    bias = np.ones((X_poly.shape[0], 1), dtype=float)
    Xb = np.concatenate([bias, X_poly], axis=1)
    y = df[target_col].to_numpy(dtype=float)
    return Xb, y

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Squared Error."""
    return float(np.mean((y_true - y_pred) ** 2))

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Root Mean Squared Error."""
    return float(np.sqrt(mse(y_true, y_pred)))

def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate R-squared score."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

def main():
    """Main function to load model, predict, and evaluate."""
    args = parse_args()
    
    _, _, results_dir = get_paths()

    # Load the trained model
    with open(args.model_path, "rb") as f:
        model = pickle.load(f)

    # Extract model parameters
    weights = np.asarray(model["weights"])
    degree = int(model["degree"])
    base_features = list(model["base_feature_names"])
    target_col = model["target_col"]

    # Load and process the dataset
    data_df = pd.read_csv(args.data_path)
    
    # Apply the same preprocessing steps as in training
    df = encode_status(data_df)
    df = impute_numeric_median(df)
    df = one_hot_encode_country(df)
    
    corr_df = load_target_correlations(results_dir)
    
    df = drop_low_correlation_features(df, corr_df, 0.05)
    df = create_interaction_features(df, corr_df, 6)
    
    df = normalize_features_to_unit_range(df)
    df = impute_numeric_median(df)


    X, y_true = df_to_design_matrix_poly(df, base_features, degree, target_col)

    # Make predictions
    y_pred = X @ weights

    # Calculate metrics
    metrics = {
        "Mean Squared Error (MSE)": mse(y_true, y_pred),
        "Root Mean Squared Error (RMSE)": rmse(y_true, y_pred),
        "R-squared (R²) Score": r2_score(y_true, y_pred),
    }

    # Save metrics to a file
    with open(args.metrics_output_path, "w", encoding="utf-8") as f:
        f.write("Regression Metrics:\n")
        for name, value in metrics.items():
            f.write(f"{name}: {value:.2f}\n")

    # Save predictions to a CSV file
    pd.DataFrame(y_pred).to_csv(args.predictions_output_path, header=False, index=False)

    print(f"Predictions saved to {args.predictions_output_path}")
    print(f"Metrics saved to {args.metrics_output_path}")

if __name__ == "__main__":
    main()