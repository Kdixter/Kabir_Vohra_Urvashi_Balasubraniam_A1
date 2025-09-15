import os
import pickle
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

# hyper paramaters:
TARGET_COL = "Life expectancy "
RANDOM_STATE = 42
VAL_FRACTION = 0.1  # fraction of train used for validation during LR sweep
EPOCHS = 3000
PATIENCE = 200
LR_CANDIDATES = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
WEIGHT_DECAY = 0.0  # L2 regularization strength (0 = off)


np.random.seed(RANDOM_STATE)


def get_paths() -> Tuple[Path, Path, Path]:
	"""Return (project_dir, data_dir, models_dir)."""
	src_dir = Path(__file__).resolve().parent
	project_dir = src_dir.parent
	data_dir = project_dir / "data"
	models_dir = project_dir / "models"
	models_dir.mkdir(parents=True, exist_ok=True)
	return project_dir, data_dir, models_dir


def load_processed_data(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
	train_path = data_dir / "life_expectancy_train_processed.csv"
	test_path = data_dir / "life_expectancy_test_processed.csv"
	if not train_path.exists() or not test_path.exists():
		raise FileNotFoundError("Processed train/test CSVs not found. Run data_preprocessing.py first.")
	return pd.read_csv(train_path), pd.read_csv(test_path)


def train_val_split(df: pd.DataFrame, val_fraction: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
	if val_fraction <= 0 or val_fraction >= 1:
		return df, df.iloc[0:0].copy()
	n = len(df)
	n_val = max(1, int(round(n * val_fraction)))
	shuffled = df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
	val_df = shuffled.iloc[:n_val].reset_index(drop=True)
	train_df = shuffled.iloc[n_val:].reset_index(drop=True)
	return train_df, val_df


def df_to_design_matrix(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
	X = df[feature_cols].to_numpy(dtype=float)
	y = df[TARGET_COL].to_numpy(dtype=float)
	# add bias term
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


def predict(Xb: np.ndarray, w: np.ndarray) -> np.ndarray:
	return Xb @ w


def compute_gradients(Xb: np.ndarray, y: np.ndarray, w: np.ndarray, weight_decay: float) -> np.ndarray:
	# gradient of MSE wrt w: (2/N) X^T (X w - y) + 2*lambda*w (except bias term not regularized)
	N = Xb.shape[0]
	y_pred = Xb @ w
	residual = y_pred - y
	grad = (2.0 / N) * (Xb.T @ residual)
	if weight_decay > 0.0:
		reg = 2.0 * weight_decay * w
		reg[0] = 0.0  # do not regularize bias
		grad = grad + reg
	return grad


def fit_gd(Xb: np.ndarray, y: np.ndarray, Xb_val: np.ndarray, y_val: np.ndarray, lr: float, epochs: int, patience: int, weight_decay: float) -> Tuple[np.ndarray, Dict[str, float]]:
	w = np.zeros(Xb.shape[1], dtype=float)
	best_w = w.copy()
	best_val = float("inf")
	stale = 0
	for epoch in range(1, epochs + 1):
		grad = compute_gradients(Xb, y, w, weight_decay)
		w = w - lr * grad
		if Xb_val.size > 0:
			y_val_pred = predict(Xb_val, w)
			val_mse = mse(y_val, y_val_pred)
			if val_mse + 1e-10 < best_val:
				best_val = val_mse
				best_w = w.copy()
				stale = 0
			else:
				stale += 1
			if stale >= patience:
				break
	return (best_w if Xb_val.size > 0 else w), {"best_val_mse": (best_val if Xb_val.size > 0 else mse(y, predict(Xb, w))), "epochs_run": epoch}


def main() -> None:
	_, data_dir, models_dir = get_paths()
	print(f"Data: {data_dir}")
	print(f"Models: {models_dir}")

	train_df, test_df = load_processed_data(data_dir)
	# Identify feature columns: all numeric except target
	feature_cols = [c for c in train_df.columns if c != TARGET_COL]

	# Train/val split for LR selection
	inner_train_df, val_df = train_val_split(train_df, VAL_FRACTION)
	Xb_train, y_train = df_to_design_matrix(inner_train_df, feature_cols)
	Xb_val, y_val = df_to_design_matrix(val_df, feature_cols) if len(val_df) else (np.empty((0, Xb_train.shape[1])), np.empty((0,)))

	# Sweep learning rates
	results = []
	for lr in LR_CANDIDATES:
		w_lr, info = fit_gd(Xb_train, y_train, Xb_val, y_val, lr=lr, epochs=EPOCHS, patience=PATIENCE, weight_decay=WEIGHT_DECAY)
		# Evaluate on val (or train if no val)
		y_eval_pred = predict(Xb_val if len(y_val) else Xb_train, w_lr)
		y_eval_true = y_val if len(y_val) else y_train
		val_mse_score = mse(y_eval_true, y_eval_pred)
		results.append({"lr": lr, "val_mse": val_mse_score, "epochs": info["epochs_run"]})
		print(f"LR {lr:.5f}: val MSE={val_mse_score:.4f}, epochs={info['epochs_run']}")

	# Choose best LR
	best = min(results, key=lambda r: r["val_mse"]) if results else {"lr": 1e-3}
	best_lr = best["lr"]
	print(f"Selected learning rate: {best_lr}")

	# Retrain on full training data with best LR
	Xb_full, y_full = df_to_design_matrix(train_df, feature_cols)
	w_final, info_final = fit_gd(Xb_full, y_full, np.empty((0, Xb_full.shape[1])), np.empty((0,)), lr=best_lr, epochs=EPOCHS, patience=PATIENCE, weight_decay=WEIGHT_DECAY)

	# Evaluate on train and test
	y_train_pred = predict(Xb_full, w_final)
	train_mse = mse(y_full, y_train_pred)
	train_mae = mae(y_full, y_train_pred)
	train_r2 = r2_score(y_full, y_train_pred)

	Xb_test, y_test = df_to_design_matrix(test_df, feature_cols)
	y_test_pred = predict(Xb_test, w_final)
	test_mse = mse(y_test, y_test_pred)
	test_mae = mae(y_test, y_test_pred)
	test_r2 = r2_score(y_test, y_test_pred)

	print("Training metrics:")
	print(f"  MSE: {train_mse:.4f}  MAE: {train_mae:.4f}  R2: {train_r2:.4f}")
	print("Test metrics:")
	print(f"  MSE: {test_mse:.4f}  MAE: {test_mae:.4f}  R2: {test_r2:.4f}")

	# Save model
	model = {
		"weights": w_final,
		"feature_names": ["__bias__"] + feature_cols,
		"target_col": TARGET_COL,
		"best_lr": best_lr,
		"train_metrics": {"mse": train_mse, "mae": train_mae, "r2": train_r2},
		"test_metrics": {"mse": test_mse, "mae": test_mae, "r2": test_r2},
	}
	model_path = models_dir / "life_expectancy_linear_model.pkl"
	with open(model_path, "wb") as f:
		pickle.dump(model, f)
	print(f"Saved model to {model_path}")


if __name__ == "__main__":
	main()
