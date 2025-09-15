import os
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


TARGET_COL = "Price"
TEST_SIZE = 0.075
RANDOM_STATE = 42
LOW_CORR_THRESHOLD = 0.03
TOP_K_FOR_INTERACTIONS = 8


def get_paths() -> Tuple[Path, Path, Path]:
	src_dir = Path(__file__).resolve().parent
	project_dir = src_dir.parent
	data_dir = project_dir / "data"
	results_dir = project_dir / "results"
	results_dir.mkdir(parents=True, exist_ok=True)
	return project_dir, data_dir, results_dir


def load_raw_dataset(data_dir: Path) -> pd.DataFrame:
	csv_path = data_dir / "Laptop Price.csv"
	if not csv_path.exists():
		raise FileNotFoundError(f"Expected data at {csv_path}")
	return pd.read_csv(csv_path)


def drop_missing_target(df: pd.DataFrame) -> pd.DataFrame:
	return df.dropna(subset=[TARGET_COL]).reset_index(drop=True)


def impute_numeric_median(df: pd.DataFrame) -> pd.DataFrame:
	for col in df.columns:
		if pd.api.types.is_numeric_dtype(df[col]) and df[col].isna().any():
			df[col] = df[col].fillna(df[col].median())
	return df


# --- Feature engineering helpers ---

def parse_screen_resolution(df: pd.DataFrame) -> pd.DataFrame:
	text_parts: List[str] = []
	x_list: List[float] = []
	y_list: List[float] = []
	pattern = re.compile(r"(\d+)\s*x\s*(\d+)")
	for val in df["ScreenResolution"].astype(str):
		matches = pattern.findall(val)
		if matches:
			x_res, y_res = matches[-1]
			x_list.append(float(x_res))
			y_list.append(float(y_res))
			prefix = val[: val.rfind(matches[-1][0] + "x" + matches[-1][1])].strip()
			text_parts.append(prefix if prefix else "")
		else:
			x_list.append(np.nan)
			y_list.append(np.nan)
			text_parts.append(val)
	df["x_res"] = x_list
	df["y_res"] = y_list
	df["ScreenText"] = [t.strip().replace("/", " ").replace("  ", " ") for t in text_parts]
	return df


def parse_cpu(df: pd.DataFrame) -> pd.DataFrame:
	freq = []
	cpu_text = []
	for val in df["Cpu"].astype(str):
		m = re.search(r"([0-9.]+)\s*GHz", val, flags=re.IGNORECASE)
		if m:
			ghz = float(m.group(1))
			freq.append(ghz)
			base = val[: m.start()].strip()
			cpu_text.append(base)
		else:
			freq.append(np.nan)
			cpu_text.append(val)
	df["cpu_ghz"] = freq
	df["cpu_text"] = cpu_text
	return df


def parse_ram_to_tb(df: pd.DataFrame) -> pd.DataFrame:
	vals_tb = []
	for val in df["Ram" ].astype(str):
		m = re.search(r"(\d+(?:\.\d+)?)\s*(TB|GB)", val, flags=re.IGNORECASE)
		if m:
			num = float(m.group(1))
			unit = m.group(2).upper()
			vals_tb.append(num if unit == "TB" else num / 1024.0)
		else:
			vals_tb.append(np.nan)
	df["ram_tb"] = vals_tb
	return df


def parse_memory(df: pd.DataFrame) -> pd.DataFrame:
	# Extract SSD total capacity in TB and a coarse memory type string
	ssd_tb = []
	mem_type = []
	for val in df["Memory"].astype(str):
		parts = [p.strip() for p in val.split("+")]
		ssd_total_tb = 0.0
		types = []
		for p in parts:
			m = re.search(r"(\d+(?:\.\d+)?)\s*(TB|GB)\s*([A-Za-z ]+)", p)
			if not m:
				continue
			num = float(m.group(1))
			unit = m.group(2).upper()
			kind = m.group(3).strip()
			amount_tb = num if unit == "TB" else num / 1024.0
			if "SSD" in kind.upper():
				ssd_total_tb += amount_tb
			types.append(kind)
		mem_type.append(" + ".join(sorted(set(types))) if types else "Unknown")
		ssd_tb.append(ssd_total_tb if ssd_total_tb > 0 else 0.0)
	df["ssd_tb"] = ssd_tb
	df["MemoryText"] = mem_type
	return df


def parse_weight(df: pd.DataFrame) -> pd.DataFrame:
	kg_vals = []
	extra_tokens = []
	for val in df["Weight"].astype(str):
		parts = [p.strip() for p in val.split(",")]
		w = parts[0]
		m = re.search(r"([0-9.]+)\s*kg", w, flags=re.IGNORECASE)
		kg_vals.append(float(m.group(1)) if m else np.nan)
		extra_tokens.append(parts[1] if len(parts) > 1 else "")
	df["weight_kg"] = kg_vals
	df["weight_extra"] = extra_tokens
	return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
	# Screen resolution parsing
	df = parse_screen_resolution(df)
	# CPU parsing
	df = parse_cpu(df)
	# RAM in TB
	df = parse_ram_to_tb(df)
	# Memory parsing (SSD only numeric)
	df = parse_memory(df)
	# Weight parsing
	df = parse_weight(df)

	# One-hot encodings
	categoricals = {
		"Company": df.get("Company"),
		"TypeName": df.get("TypeName"),
		"ScreenText": df.get("ScreenText"),
		"cpu_text": df.get("cpu_text"),
		"MemoryText": df.get("MemoryText"),
		"Gpu": df.get("Gpu"),
		"OpSys": df.get("OpSys"),
		"weight_extra": df.get("weight_extra"),
	}
	for name, series in categoricals.items():
		if series is not None:
			dummies = pd.get_dummies(series.astype(str), prefix=name)
			df = pd.concat([df, dummies], axis=1)

	# Numeric keepers
	numeric_keep = [
		"Inches", "x_res", "y_res", "cpu_ghz", "ram_tb", "ssd_tb", "weight_kg"
	]
	# Drop original raw categorical columns that have been encoded
	drop_cols = [
		"ScreenResolution", "ScreenText", "Cpu", "cpu_text", "Ram", "Memory", "MemoryText",
		"Gpu", "OpSys", "Weight", "weight_extra", "Company", "TypeName"
	]
	for c in drop_cols:
		if c in df.columns:
			df = df.drop(columns=[c])

	# Ensure numeric types are numeric
	for col in numeric_keep:
		if col in df.columns:
			df[col] = pd.to_numeric(df[col], errors="coerce")

	return df


def normalize_features_to_unit_range(df: pd.DataFrame) -> pd.DataFrame:
	for col in df.columns:
		if col == TARGET_COL:
			continue
		if pd.api.types.is_bool_dtype(df[col]):
			df[col] = df[col].astype(int)
		if pd.api.types.is_numeric_dtype(df[col]):
			mn = df[col].min()
			mx = df[col].max()
			if pd.isna(mn) or pd.isna(mx):
				continue
			if mx != mn:
				df[col] = 2 * (df[col] - mn) / (mx - mn) - 1
	return df


def load_target_correlations(results_dir: Path) -> pd.DataFrame:
	corr_path = results_dir / "feature_price_correlations.csv"
	if corr_path.exists():
		corr_df = pd.read_csv(corr_path)
		if set(corr_df.columns) >= {"feature", "pearson_corr_with_target"}:
			return corr_df
	# fallback: compute from numeric subset
	print("feature_price_correlations.csv not found or invalid; recomputing from numeric subset.")
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
		val = float(row["pearson_corr_with_target"])
		if abs(val) < threshold and feat in df.columns:
			to_drop.append(feat)
	if to_drop:
		print(f"Dropping low-correlation features (|r|<{threshold}): {len(to_drop)}")
		df = df.drop(columns=to_drop)
	return df


def create_interaction_features(df: pd.DataFrame, corr_df: pd.DataFrame, top_k: int) -> pd.DataFrame:
	corr_df = corr_df[corr_df["feature"] != TARGET_COL].copy()
	corr_df["abs_corr"] = corr_df["pearson_corr_with_target"].abs()
	top_feats = [f for f in corr_df.sort_values("abs_corr", ascending=False)["feature"].tolist() if f in df.columns][:top_k]
	created = 0
	for i, a in enumerate(top_feats):
		for b in top_feats[i + 1:]:
			new_name = f"{a}__x__{b}"
			if pd.api.types.is_numeric_dtype(df[a]) and pd.api.types.is_numeric_dtype(df[b]):
				df[new_name] = df[a] * df[b]
				created += 1
	print(f"Created {created} interaction features from top {len(top_feats)} features")
	return df


def split_and_save(df: pd.DataFrame, data_dir: Path) -> Tuple[Path, Path]:
	df_shuffled = df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
	n = len(df_shuffled)
	n_test = int(round(n * TEST_SIZE))
	test_df = df_shuffled.iloc[:n_test].reset_index(drop=True)
	train_df = df_shuffled.iloc[n_test:].reset_index(drop=True)
	train_path = data_dir / "laptop_train_processed.csv"
	test_path = data_dir / "laptop_test_processed.csv"
	train_df.to_csv(train_path, index=False)
	test_df.to_csv(test_path, index=False)
	print(f"Saved train ({len(train_df)}) to {train_path} and test ({len(test_df)}) to {test_path}")
	return train_path, test_path


def main() -> None:
	_, data_dir, results_dir = get_paths()
	df = load_raw_dataset(data_dir)
	df = drop_missing_target(df)

	# Feature engineering
	df = engineer_features(df)

	# Bring target back if dropped from engineering
	if TARGET_COL not in df.columns:
		raise RuntimeError("Target column missing after engineering")

	# Impute numerics
	df = impute_numeric_median(df)

	# Drop low-corr original numeric features where applicable
	corr_df = load_target_correlations(results_dir)
	df = drop_low_correlation_features(df, corr_df, LOW_CORR_THRESHOLD)

	# Interactions from top correlated base features
	df = create_interaction_features(df, corr_df, TOP_K_FOR_INTERACTIONS)

	# Normalize predictors to [-1, 1]
	df = normalize_features_to_unit_range(df)

	# Safety impute
	df = impute_numeric_median(df)

	# Split/save
	split_and_save(df, data_dir)


if __name__ == "__main__":
	main()
