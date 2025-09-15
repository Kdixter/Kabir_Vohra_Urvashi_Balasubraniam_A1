import pickle
from itertools import combinations_with_replacement
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd


TARGET_COL = "Life expectancy "


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


def df_to_design_matrix_poly(df: pd.DataFrame, base_features: List[str], degree: int) -> Tuple[np.ndarray, np.ndarray, List[str]]:
	X_poly, term_names = expand_polynomial_features(df, base_features, degree)
	bias = np.ones((X_poly.shape[0], 1), dtype=float)
	Xb = np.concatenate([bias, X_poly], axis=1)
	y = df[TARGET_COL].to_numpy(dtype=float)
	return Xb, y, term_names


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
	model_path = models_dir / "life_expectancy_polynomial_model.pkl"
	test_path = data_dir / "life_expectancy_test_processed.csv"

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
	Xb_test, y_test, term_names = df_to_design_matrix_poly(test_df, base_features, degree)
	assert Xb_test.shape[1] == weights.shape[0], "Design matrix and weight vector size mismatch"
	y_pred = Xb_test @ weights

	metrics = {
		"mse": mse(y_test, y_pred),
		"mae": mae(y_test, y_pred),
		"r2": r2_score(y_test, y_pred),
	}
	print("Test metrics (poly):")
	print(f"  MSE: {metrics['mse']:.4f}  MAE: {metrics['mae']:.4f}  R2: {metrics['r2']:.4f}")

	pred_df = pd.DataFrame({
		"y_true": y_test,
		"y_pred": y_pred,
	})
	pred_path = results_dir / "polynomial_test_predictions.csv"
	pred_df.to_csv(pred_path, index=False)

	metrics_path = results_dir / "polynomial_test_metrics.txt"
	with open(metrics_path, "w", encoding="utf-8") as f:
		f.write(f"MSE: {metrics['mse']:.6f}\n")
		f.write(f"MAE: {metrics['mae']:.6f}\n")
		f.write(f"R2:  {metrics['r2']:.6f}\n")

	print(f"Saved predictions to {pred_path}")
	print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
	main()
