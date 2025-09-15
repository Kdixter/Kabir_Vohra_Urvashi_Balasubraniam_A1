import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


sns.set(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 120


TARGET_COL = "Life expectancy "


def get_paths() -> Tuple[Path, Path, Path]:
	"""Return (project_dir, data_dir, results_dir) based on this file's location."""
	src_dir = Path(__file__).resolve().parent
	project_dir = src_dir.parent
	data_dir = project_dir / "data"
	results_dir = project_dir / "results"
	results_dir.mkdir(parents=True, exist_ok=True)
	return project_dir, data_dir, results_dir


def load_dataset(data_dir: Path) -> pd.DataFrame:
	csv_path = data_dir / "life_expectancy.csv"
	if not csv_path.exists():
		raise FileNotFoundError(f"Expected data at {csv_path}")
	df = pd.read_csv(csv_path)
	return df


def prepare_numeric_dataframe(df: pd.DataFrame) -> pd.DataFrame:
	if TARGET_COL not in df.columns:
		raise KeyError(f"Target column '{TARGET_COL}' not found. Columns: {list(df.columns)}")

	# Keep numeric columns and ensure target is present
	numeric_df = df.select_dtypes(include=[np.number]).copy()
	if TARGET_COL not in numeric_df.columns:
		numeric_df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")

	# Drop missing target rows
	numeric_df = numeric_df.dropna(subset=[TARGET_COL]).reset_index(drop=True)

	# Impute remaining NaNs with column medians
	for col in numeric_df.columns:
		if numeric_df[col].isna().any():
			numeric_df[col] = numeric_df[col].fillna(numeric_df[col].median())

	return numeric_df


def save_target_correlations(numeric_df: pd.DataFrame, results_dir: Path) -> pd.Series:
	corr_with_target = numeric_df.corr(method="pearson")[TARGET_COL].sort_values(ascending=False)
	corr_df = corr_with_target.reset_index()
	corr_df.columns = ["feature", "pearson_corr_with_target"]
	out_path = results_dir / "feature_correlations.csv"
	corr_df.to_csv(out_path, index=False)
	print(f"Saved target correlations to {out_path}")
	return corr_with_target


def plot_top_target_correlations(abs_sorted: pd.Series, results_dir: Path, top_n: int = 15) -> List[str]:
	plot_series = abs_sorted.head(top_n)
	plt.figure(figsize=(8, 0.4 * top_n + 2))
	sns.barplot(x=plot_series.values, y=plot_series.index, palette="viridis")
	plt.title("Top correlations with Life expectancy (absolute)")
	plt.xlabel("Pearson correlation")
	plt.ylabel("Feature")
	plt.tight_layout()
	bar_path = results_dir / "top_target_correlations_bar.png"
	plt.savefig(bar_path, bbox_inches="tight")
	plt.close()
	print(f"Saved bar chart to {bar_path}")
	return [str(bar_path)]


def plot_scatter_regplots(numeric_df: pd.DataFrame, top_features: List[str], results_dir: Path) -> str:
	num_features = min(len(top_features), 6)
	top_features = top_features[:num_features]
	num_cols = 3
	num_rows = int(np.ceil(num_features / num_cols)) or 1
	fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 4 * num_rows))
	axes = np.array(axes).ravel() if isinstance(axes, np.ndarray) else np.array([axes])
	for i, feat in enumerate(top_features):
		ax = axes[i]
		sns.regplot(x=numeric_df[feat], y=numeric_df[TARGET_COL],
				scatter_kws={"alpha": 0.3, "s": 15}, line_kws={"color": "red"}, ax=ax)
		ax.set_title(f"{feat} vs {TARGET_COL}")
	# Hide any unused subplots
	for j in range(i + 1, len(axes)):
		axes[j].axis("off")
	plt.tight_layout()
	scatter_path = results_dir / "top_target_scatter_regplots.png"
	plt.savefig(scatter_path, bbox_inches="tight")
	plt.close()
	print(f"Saved scatter/regplot grid to {scatter_path}")
	return str(scatter_path)


def plot_top_heatmap(numeric_df: pd.DataFrame, abs_sorted: pd.Series, results_dir: Path, top_m: int = 12) -> str:
	heatmap_feats = list(abs_sorted.index[:top_m]) + [TARGET_COL]
	corr_mat_subset = numeric_df[heatmap_feats].corr()
	plt.figure(figsize=(1.1 * len(heatmap_feats), 0.9 * len(heatmap_feats)))
	sns.heatmap(corr_mat_subset, annot=False, cmap="coolwarm", center=0, linewidths=0.5)
	plt.title("Correlation heatmap: top features and target")
	plt.tight_layout()
	heatmap_path = results_dir / "corr_heatmap_top_features.png"
	plt.savefig(heatmap_path, bbox_inches="tight")
	plt.close()
	print(f"Saved heatmap to {heatmap_path}")
	return str(heatmap_path)


def inter_feature_analysis(numeric_df: pd.DataFrame, results_dir: Path) -> None:
	# Full correlation matrix
	corr_matrix = numeric_df.corr(method="pearson")
	corr_mat_path = results_dir / "full_correlation_matrix.csv"
	corr_matrix.to_csv(corr_mat_path)
	print(f"Saved full correlation matrix to {corr_mat_path}")

	# Heatmap for all features
	plt.figure(figsize=(12, 10))
	sns.heatmap(corr_matrix, cmap="coolwarm", center=0, cbar_kws={"shrink": 0.8})
	plt.title("Feature correlation heatmap (all numeric)")
	plt.tight_layout()
	overall_heatmap_path = results_dir / "full_correlation_heatmap.png"
	plt.savefig(overall_heatmap_path, bbox_inches="tight")
	plt.close()
	print(f"Saved full correlation heatmap to {overall_heatmap_path}")

	# High correlation pairs among predictors
	predictor_columns = [c for c in corr_matrix.columns if c != TARGET_COL]
	threshold = 0.8
	high_pairs = []
	for i, a in enumerate(predictor_columns):
		for j, b in enumerate(predictor_columns):
			if j <= i:
				continue
			r = corr_matrix.loc[a, b]
			if abs(r) >= threshold:
				high_pairs.append((a, b, r))
	
	high_corr_df = pd.DataFrame(high_pairs, columns=["feature_a", "feature_b", "pearson_r"]).sort_values(
		by="pearson_r", key=lambda s: s.abs(), ascending=False
	)
	high_pairs_path = results_dir / "high_corr_feature_pairs.csv"
	high_corr_df.to_csv(high_pairs_path, index=False)
	print(f"Saved high-correlation predictor pairs (|r|>={threshold}) to {high_pairs_path}")


def suggest_interaction_candidates(abs_sorted: pd.Series, results_dir: Path, top_k: int = 8) -> None:
	features = list(abs_sorted.index[:top_k])
	interaction_candidates = []
	for i, a in enumerate(features):
		for b in features[i + 1:]:
			interaction_candidates.append({
				"interaction": f"{a}*{b}",
				"feat_a": a,
				"feat_b": b,
				"abs_corr_a": abs_sorted.loc[a],
				"abs_corr_b": abs_sorted.loc[b],
			})
	interaction_df = pd.DataFrame(interaction_candidates)
	interactions_path = results_dir / "interaction_candidates_from_top_features.csv"
	interaction_df.to_csv(interactions_path, index=False)
	print(f"Saved interaction candidates to {interactions_path}")


def main() -> None:
	_, data_dir, results_dir = get_paths()
	print(f"Data directory: {data_dir}")
	print(f"Results directory: {results_dir}")

	df = load_dataset(data_dir)
	numeric_df = prepare_numeric_dataframe(df)

	corr_with_target = save_target_correlations(numeric_df, results_dir)
	corr_no_target = corr_with_target.drop(labels=[TARGET_COL])
	abs_sorted = corr_no_target.reindex(corr_no_target.abs().sort_values(ascending=False).index)

	# Visualizations and exports
	plot_top_target_correlations(abs_sorted, results_dir, top_n=15)
	plot_scatter_regplots(numeric_df, list(abs_sorted.index[:6]), results_dir)
	plot_top_heatmap(numeric_df, abs_sorted, results_dir, top_m=12)

	# Inter-feature correlations
	inter_feature_analysis(numeric_df, results_dir)
	# Interaction candidates
	suggest_interaction_candidates(abs_sorted, results_dir, top_k=8)


if __name__ == "__main__":
	main()
