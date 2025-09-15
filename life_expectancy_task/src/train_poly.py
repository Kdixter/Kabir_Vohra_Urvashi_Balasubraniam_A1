import os
import pickle
from itertools import combinations_with_replacement
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd


TARGET_COL = "Life expectancy "
RANDOM_STATE = 42
VAL_FRACTION = 0.1
EPOCHS = 4000
PATIENCE = 300
LR_CANDIDATES = [3e-4, 1e-3, 3e-3, 1e-2]
WEIGHT_DECAY = 0.0
POLY_DEGREE = 2  # polynomial degree 
MAX_FEATURES_FOR_EXPANSION = 150  # safety cap to avoid explosion
TOP_BASE_FEATURES_FOR_POLY = 80   # select top features by |corr| for poly expansion


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


def select_top_base_features_by_corr(train_df: pd.DataFrame, top_k: int) -> List[str]:
	"""Use precomputed correlations if available to select top_k features by |corr|.
	Fallback to variance-based ranking if file not found."""
	project_dir = Path(__file__).resolve().parent.parent
	corr_path = project_dir / "results" / "feature_correlations.csv"
	all_features = [c for c in train_df.columns if c != TARGET_COL]
	if corr_path.exists():
		corr_df = pd.read_csv(corr_path)
		if set(corr_df.columns) >= {"feature", "pearson_corr_with_target"}:
			corr_df["abs_corr"] = corr_df["pearson_corr_with_target"].abs()
			ordered = [f for f in corr_df.sort_values("abs_corr", ascending=False)["feature"].tolist() if f in all_features]
			return ordered[:top_k]
	# fallback: variance ranking
	variances = train_df[all_features].var(numeric_only=True).sort_values(ascending=False)
	ordered = [c for c in variances.index if c in all_features]
	return ordered[:top_k]


def build_poly_feature_map(base_features: List[str], degree: int) -> List[Tuple[Tuple[int, ...], Tuple[str, ...]]]:
	"""Return mapping of polynomial terms using index tuples and names.
	Each term is represented by a tuple of base feature indices (with replacement, sorted).
	For degree=1, includes each base feature once; for degree=2, includes all pairs (i<=j), etc.
	"""
	index_map = list(range(len(base_features)))
	terms: List[Tuple[Tuple[int, ...], Tuple[str, ...]]] = []
	for d in range(1, degree + 1):
		for combo in combinations_with_replacement(index_map, d):
			name_tuple = tuple(base_features[i] for i in combo)
			terms.append((combo, name_tuple))
	return terms


def expand_polynomial_features(df: pd.DataFrame, base_features: List[str], degree: int) -> Tuple[np.ndarray, List[str]]:
	if len(base_features) > MAX_FEATURES_FOR_EXPANSION:
		raise RuntimeError(f"Too many base features ({len(base_features)}). Reduce dimensionality before polynomial expansion.")
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


def predict(Xb: np.ndarray, w: np.ndarray) -> np.ndarray:
	return Xb @ w


def compute_gradients(Xb: np.ndarray, y: np.ndarray, w: np.ndarray, weight_decay: float) -> np.ndarray:
	N = Xb.shape[0]
	y_pred = Xb @ w
	residual = y_pred - y
	grad = (2.0 / N) * (Xb.T @ residual)
	if weight_decay > 0.0:
		reg = 2.0 * weight_decay * w
		reg[0] = 0.0
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
	# select top base features to keep polynomial dimensionality reasonable
	base_features_all = [c for c in train_df.columns if c != TARGET_COL]
	base_features = select_top_base_features_by_corr(train_df, TOP_BASE_FEATURES_FOR_POLY)
	print(f"Using {len(base_features)} base features out of {len(base_features_all)} for polynomial expansion")

	# train/val split
	inner_train_df, val_df = train_val_split(train_df, VAL_FRACTION)
	Xb_train, y_train, term_names = df_to_design_matrix_poly(inner_train_df, base_features, POLY_DEGREE)
	Xb_val, y_val, _ = df_to_design_matrix_poly(val_df, base_features, POLY_DEGREE) if len(val_df) else (np.empty((0, Xb_train.shape[1])), np.empty((0,)), term_names)

	# LR sweep
	results = []
	for lr in LR_CANDIDATES:
		w_lr, info = fit_gd(Xb_train, y_train, Xb_val, y_val, lr=lr, epochs=EPOCHS, patience=PATIENCE, weight_decay=WEIGHT_DECAY)
		y_eval_pred = predict(Xb_val if len(y_val) else Xb_train, w_lr)
		y_eval_true = y_val if len(y_val) else y_train
		val_mse_score = mse(y_eval_true, y_eval_pred)
		results.append({"lr": lr, "val_mse": val_mse_score, "epochs": info["epochs_run"]})
		print(f"LR {lr:.5f}: val MSE={val_mse_score:.4f}, epochs={info['epochs_run']}")

	best = min(results, key=lambda r: r["val_mse"]) if results else {"lr": 1e-3}
	best_lr = best["lr"]
	print(f"Selected learning rate: {best_lr}")

	# retrain on full train
	Xb_full, y_full, term_names_full = df_to_design_matrix_poly(train_df, base_features, POLY_DEGREE)
	w_final, info_final = fit_gd(Xb_full, y_full, np.empty((0, Xb_full.shape[1])), np.empty((0,)), lr=best_lr, epochs=EPOCHS, patience=PATIENCE, weight_decay=WEIGHT_DECAY)

	# metrics
	y_train_pred = predict(Xb_full, w_final)
	train_mse = mse(y_full, y_train_pred)
	train_mae = mae(y_full, y_train_pred)
	train_r2 = r2_score(y_full, y_train_pred)

	Xb_test, y_test, _ = df_to_design_matrix_poly(test_df, base_features, POLY_DEGREE)
	y_test_pred = predict(Xb_test, w_final)
	test_mse = mse(y_test, y_test_pred)
	test_mae = mae(y_test, y_test_pred)
	test_r2 = r2_score(y_test, y_test_pred)

	print("Training metrics (poly):")
	print(f"  MSE: {train_mse:.4f}  MAE: {train_mae:.4f}  R2: {train_r2:.4f}")
	print("Test metrics (poly):")
	print(f"  MSE: {test_mse:.4f}  MAE: {test_mae:.4f}  R2: {test_r2:.4f}")

	# save model
	model = {
		"weights": w_final,
		"degree": POLY_DEGREE,
		"base_feature_names": base_features,
		"poly_term_names": ["__bias__"] + term_names_full,
		"target_col": TARGET_COL,
		"best_lr": best_lr,
		"train_metrics": {"mse": train_mse, "mae": train_mae, "r2": train_r2},
		"test_metrics": {"mse": test_mse, "mae": test_mae, "r2": test_r2},
	}
	model_path = models_dir / "life_expectancy_polynomial_model.pkl"
	with open(model_path, "wb") as f:
		pickle.dump(model, f)
	print(f"Saved polynomial model to {model_path}")


if __name__ == "__main__":
	main()
