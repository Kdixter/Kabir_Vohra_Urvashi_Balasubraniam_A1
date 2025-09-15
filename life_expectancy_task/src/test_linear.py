import pickle
from pathlib import Path
from typing import Tuple

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


def df_to_design_matrix(df: pd.DataFrame, feature_cols: list) -> Tuple[np.ndarray, np.ndarray]:
	X = df[feature_cols].to_numpy(dtype=float)
	y = df[TARGET_COL].to_numpy(dtype=float)
	bias = np.ones((X.shape[0], 1), dtype=float)
	Xb = np.concatenate([bias, X], axis=1)
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
	model_path = models_dir / "life_expectancy_linear_model.pkl"
	test_path = data_dir / "life_expectancy_test_processed.csv"

	if not model_path.exists():
		raise FileNotFoundError(f"Model not found at {model_path}. Train it first.")
	if not test_path.exists():
		raise FileNotFoundError(f"Test data not found at {test_path}. Run preprocessing first.")

	with open(model_path, "rb") as f:
		model = pickle.load(f)

	feature_names = model["feature_names"]
	assert feature_names[0] == "__bias__", "First feature must be bias placeholder"
	feature_cols = feature_names[1:]
	weights = np.asarray(model["weights"], dtype=float)

	test_df = pd.read_csv(test_path)
	Xb_test, y_test = df_to_design_matrix(test_df, feature_cols)
	y_pred = Xb_test @ weights

	metrics = {
		"mse": mse(y_test, y_pred),
		"mae": mae(y_test, y_pred),
		"r2": r2_score(y_test, y_pred),
	}
	print("Test metrics:")
	print(f"  MSE: {metrics['mse']:.4f}  MAE: {metrics['mae']:.4f}  R2: {metrics['r2']:.4f}")

	# Save predictions and metrics
	pred_df = pd.DataFrame({
		"y_true": y_test,
		"y_pred": y_pred,
	})
	pred_path = results_dir / "linear_test_predictions.csv"
	pred_df.to_csv(pred_path, index=False)

	metrics_path = results_dir / "linear_test_metrics.txt"
	with open(metrics_path, "w", encoding="utf-8") as f:
		f.write(f"MSE: {metrics['mse']:.6f}\n")
		f.write(f"MAE: {metrics['mae']:.6f}\n")
		f.write(f"R2:  {metrics['r2']:.6f}\n")

	print(f"Saved predictions to {pred_path}")
	print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
	main()
