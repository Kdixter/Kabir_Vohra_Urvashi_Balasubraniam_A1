import pickle
from itertools import combinations_with_replacement
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd


TARGET_COL = "Price"


def get_paths() -> Tuple[Path, Path, Path]:
	src_dir = Path(__file__).resolve().parent
	project_dir = src_dir.parent
	data_dir = project_dir / "data"
	results_dir = project_dir / "results"
	results_dir.mkdir(parents=True, exist_ok=True)
	models_dir = project_dir / "models"
	return data_dir, results_dir, models_dir


def build_poly_feature_map(base_features: List[str], degree: int) -> List[tuple]:
	index_map = list(range(len(base_features)))
	terms = []
	for d in range(1, degree + 1):
		for combo in combinations_with_replacement(index_map, d):
			name_tuple = tuple(base_features[i] for i in combo)
			terms.append((combo, name_tuple))
	return terms


def expand_polynomial_features(df: pd.DataFrame, base_features: List[str], degree: int) -> Tuple[np.ndarray, List[str]]:
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


def df_to_design_matrix_poly(df: pd.DataFrame, base_features: List[str], degree: int) -> Tuple[np.ndarray, np.ndarray]:
	X_poly, term_names = expand_polynomial_features(df, base_features, degree)
	bias = np.ones((X_poly.shape[0], 1), dtype=float)
	Xb = np.concatenate([bias, X_poly], axis=1)
	y = df[TARGET_COL].to_numpy(dtype=float)
	return Xb, y


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
	return float(np.mean((y_true - y_pred) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
	return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
	ss_res = np.sum((y_true - y_pred) ** 2)
	ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
	return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def main() -> None:
	data_dir, results_dir, models_dir = get_paths()
	model_path = models_dir / "laptop_polynomial_log_model.pkl"
	test_path = data_dir / "laptop_test_processed.csv"

	if not model_path.exists():
		raise FileNotFoundError(f"Model not found at {model_path}. Train it first.")
	if not test_path.exists():
		raise FileNotFoundError(f"Test data not found at {test_path}. Run preprocessing first.")

	with open(model_path, "rb") as f:
		model = pickle.load(f)

	degree = int(model["degree"])
	base_features: List[str] = list(model["base_feature_names"])  # order matters
	weights = np.asarray(model["weights"], dtype=float)

	test_df = pd.read_csv(test_path)
	Xb_test, y_test_price = df_to_design_matrix_poly(test_df, base_features, degree)

	# Predict log-price, then back-transform to price
	y_pred_log = Xb_test @ weights
	y_pred_price = np.exp(y_pred_log)
	y_test_log = np.log(np.clip(y_test_price, 1e-6, None))

	metrics_log = {
		"mse": mse(y_test_log, y_pred_log),
		"mae": mae(y_test_log, y_pred_log),
		"r2": r2_score(y_test_log, y_pred_log),
	}
	metrics_price = {
		"mse": mse(y_test_price, y_pred_price),
		"mae": mae(y_test_price, y_pred_price),
		"r2": r2_score(y_test_price, y_pred_price),
	}

	print("Laptop test metrics (poly, log-space):")
	print(f"  MSE: {metrics_log['mse']:.6f}  MAE: {metrics_log['mae']:.6f}  R2: {metrics_log['r2']:.6f}")
	print("Laptop test metrics (poly, price-space):")
	print(f"  MSE: {metrics_price['mse']:.2f}  MAE: {metrics_price['mae']:.2f}  R2: {metrics_price['r2']:.4f}")

	pred_df = pd.DataFrame({
		"y_true_price": y_test_price,
		"y_pred_price": y_pred_price,
		"y_true_log": y_test_log,
		"y_pred_log": y_pred_log,
	})
	pred_path = results_dir / "laptop_polynomial_log_test_predictions.csv"
	pred_df.to_csv(pred_path, index=False)

	metrics_path = results_dir / "laptop_polynomial_log_test_metrics.txt"
	with open(metrics_path, "w", encoding="utf-8") as f:
		f.write("LOG-SPACE\n")
		f.write(f"MSE: {metrics_log['mse']:.6f}\n")
		f.write(f"MAE: {metrics_log['mae']:.6f}\n")
		f.write(f"R2:  {metrics_log['r2']:.6f}\n")
		f.write("\nPRICE-SPACE\n")
		f.write(f"MSE: {metrics_price['mse']:.6f}\n")
		f.write(f"MAE: {metrics_price['mae']:.6f}\n")
		f.write(f"R2:  {metrics_price['r2']:.6f}\n")

	print(f"Saved predictions to {pred_path}")
	print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
	main()
