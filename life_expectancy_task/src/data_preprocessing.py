import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


TARGET_COL = "Life expectancy "
STATUS_COL = "Status"
COUNTRY_COL = "Country"

# hyper paramaters for data enginerring: 
LOW_CORR_THRESHOLD = 0.05  # drop features with |corr| below this
TOP_K_FOR_INTERACTIONS = 6  # create pairwise products among these many top features (expiremented with 4,5,6,7,9)
RANDOM_STATE = 42 # for randomness
TEST_SIZE = 0.075  


def get_paths() -> Tuple[Path, Path, Path]:
	src_dir = Path(__file__).resolve().parent
	project_dir = src_dir.parent
	data_dir = project_dir / "data"
	results_dir = project_dir / "results"
	results_dir.mkdir(parents=True, exist_ok=True)
	return project_dir, data_dir, results_dir


def load_raw_dataset(data_dir: Path) -> pd.DataFrame:
	csv_path = data_dir / "life_expectancy.csv"
	if not csv_path.exists():
		raise FileNotFoundError(f"Expected data at {csv_path}")
	return pd.read_csv(csv_path)


def drop_missing_target(df: pd.DataFrame) -> pd.DataFrame:
	return df.dropna(subset=[TARGET_COL]).reset_index(drop=True)


def encode_status(df: pd.DataFrame) -> pd.DataFrame:
	if STATUS_COL in df.columns:
		mapping = {"Developed": 0.5, "Developing": -0.25}
		df[STATUS_COL] = df[STATUS_COL].map(mapping)
	return df


def impute_numeric_median(df: pd.DataFrame) -> pd.DataFrame:
	for col in df.columns:
		if pd.api.types.is_numeric_dtype(df[col]):
			if df[col].isna().any():
				df[col] = df[col].fillna(df[col].median())
	return df


def one_hot_encode_country(df: pd.DataFrame) -> pd.DataFrame:
	if COUNTRY_COL in df.columns:
		dummies = pd.get_dummies(df[COUNTRY_COL], prefix=COUNTRY_COL)
		df = pd.concat([df.drop(columns=[COUNTRY_COL]), dummies], axis=1)
	return df


def normalize_features_to_unit_range(df: pd.DataFrame) -> pd.DataFrame:
	
	for col in df.columns:
		if col == TARGET_COL:
			continue
		# Cast boolean dtypes to integers to allow arithmetic
		if pd.api.types.is_bool_dtype(df[col]):
			df[col] = df[col].astype(int)
		if pd.api.types.is_numeric_dtype(df[col]):
			col_min = df[col].min()
			col_max = df[col].max()
			if pd.isna(col_min) or pd.isna(col_max):
				continue
			if col_max != col_min:
				df[col] = 2 * (df[col] - col_min) / (col_max - col_min) - 1
			# if constant, leave as is
	return df


def load_target_correlations(results_dir: Path) -> pd.DataFrame:
	corr_path = results_dir / "feature_correlations.csv"
	if corr_path.exists():
		corr_df = pd.read_csv(corr_path)
		# ensure columns
		if set(corr_df.columns) >= {"feature", "pearson_corr_with_target"}:
			return corr_df
	# Fallback: compute from raw numeric
	print("feature_correlations.csv not found or invalid; recomputing from raw data.")
	_, data_dir, _ = get_paths()
	raw = load_raw_dataset(data_dir)
	numeric = raw.select_dtypes(include=[np.number]).copy()
	if TARGET_COL not in numeric.columns and TARGET_COL in raw.columns:
		numeric[TARGET_COL] = pd.to_numeric(raw[TARGET_COL], errors="coerce")
	numeric = numeric.dropna(subset=[TARGET_COL])
	for col in numeric.columns:
		if numeric[col].isna().any():
			numeric[col] = numeric[col].fillna(numeric[col].median())
	series = numeric.corr(method="pearson")[TARGET_COL].sort_values(ascending=False)
	corr_df = series.reset_index()
	corr_df.columns = ["feature", "pearson_corr_with_target"]
	return corr_df


def drop_low_correlation_features(df: pd.DataFrame, corr_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
	
	to_drop: List[str] = []
	for _, row in corr_df.iterrows():
		feat = row["feature"]
		if feat == TARGET_COL:
			continue
		corr_val = float(row["pearson_corr_with_target"])
		if abs(corr_val) < threshold and feat in df.columns:
			to_drop.append(feat)
	if to_drop:
		print(f"Dropping low-correlation features (|r|<{threshold}): {len(to_drop)}")
		df = df.drop(columns=to_drop)
	return df


def create_interaction_features(df: pd.DataFrame, corr_df: pd.DataFrame, top_k: int) -> pd.DataFrame:
	
	# order by absolute correlation excluding target
	corr_df_filtered = corr_df[corr_df["feature"] != TARGET_COL].copy()
	corr_df_filtered["abs_corr"] = corr_df_filtered["pearson_corr_with_target"].abs()
	top_feats = [f for f in corr_df_filtered.sort_values("abs_corr", ascending=False)["feature"].tolist() if f in df.columns][:top_k]
	created = 0
	for i, a in enumerate(top_feats):
		for b in top_feats[i + 1:]:
			new_name = f"{a}__x__{b}"
			if a in df.columns and b in df.columns and pd.api.types.is_numeric_dtype(df[a]) and pd.api.types.is_numeric_dtype(df[b]):
				df[new_name] = df[a] * df[b]
				created += 1
	print(f"Created {created} interaction features from top {len(top_feats)} features")
	return df


def split_and_save(df: pd.DataFrame, data_dir: Path, test_size: float, random_state: int) -> Tuple[Path, Path]:
	df_shuffled = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
	n_total = len(df_shuffled)
	n_test = int(round(n_total * test_size))
	test_df = df_shuffled.iloc[:n_test].reset_index(drop=True)
	train_df = df_shuffled.iloc[n_test:].reset_index(drop=True)
	train_path = data_dir / "life_expectancy_train_processed.csv"
	test_path = data_dir / "life_expectancy_test_processed.csv"
	train_df.to_csv(train_path, index=False)
	test_df.to_csv(test_path, index=False)
	print(f"Saved train to {train_path} ({len(train_df)} rows) and test to {test_path} ({len(test_df)} rows)")
	return train_path, test_path

# implementingf all functions to my data: 
def main() -> None:
	_, data_dir, results_dir = get_paths()
	print(f"Data directory: {data_dir}")
	print(f"Results directory: {results_dir}")

	# Load and basic cleaning
	df = load_raw_dataset(data_dir)
	df = drop_missing_target(df)
	df = encode_status(df)

	df = impute_numeric_median(df)

	
	df = one_hot_encode_country(df) # One-hot encode country

	
	corr_df = load_target_correlations(results_dir)
	df = drop_low_correlation_features(df, corr_df, LOW_CORR_THRESHOLD)

	
	df = create_interaction_features(df, corr_df, TOP_K_FOR_INTERACTIONS)

	
	df = normalize_features_to_unit_range(df)

	
	df = impute_numeric_median(df) # Final median imputation safeguard (in case interactions introduced NaNs)

	
	split_and_save(df, data_dir, TEST_SIZE, RANDOM_STATE) # Split and save


if __name__ == "__main__":
	main()
