import os
import pandas as pd
from typing import Dict, List


def read_results_file(file_path: str) -> Dict[str, float]:
    """
    Read results from a text file and extract metrics.
    
    Args:
        file_path: Path to the results file
        
    Returns:
        Dictionary containing metrics
    """
    metrics = {}
    
    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}")
        return metrics
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if 'Mean Squared Error (MSE):' in line:
            metrics['MSE'] = float(line.split(':')[1].strip())
        elif 'Root Mean Squared Error (RMSE):' in line:
            metrics['RMSE'] = float(line.split(':')[1].strip())
        elif 'Mean Absolute Error (MAE):' in line:
            metrics['MAE'] = float(line.split(':')[1].strip())
        elif 'R-squared:' in line:
            metrics['R_squared'] = float(line.split(':')[1].strip())
        elif 'Mean Absolute Percentage Error (MAPE):' in line:
            mape_str = line.split(':')[1].strip().replace('%', '')
            if mape_str != 'inf%':
                metrics['MAPE'] = float(mape_str)
            else:
                metrics['MAPE'] = float('inf')
    
    return metrics


def create_comparison_table(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Create a comparison table from results dictionary.
    
    Args:
        results: Dictionary with model names as keys and metrics as values
        
    Returns:
        DataFrame with comparison results
    """
    # Create DataFrame
    df = pd.DataFrame(results).T
    
    # Reorder columns for better readability
    column_order = ['MSE', 'RMSE', 'MAE', 'R_squared', 'MAPE']
    df = df[column_order]
    
    # Round to appropriate decimal places
    df['MSE'] = df['MSE'].round(4)
    df['RMSE'] = df['RMSE'].round(4)
    df['MAE'] = df['MAE'].round(4)
    df['R_squared'] = df['R_squared'].round(4)
    df['MAPE'] = df['MAPE'].round(2)
    
    return df


def print_analysis(df: pd.DataFrame):
    """
    Print detailed analysis of model comparison.
    
    Args:
        df: DataFrame with model comparison results
    """
    print("\n" + "=" * 80)
    print("DETAILED MODEL ANALYSIS")
    print("=" * 80)
    
    # Find best performing model for each metric
    print("\n🏆 BEST PERFORMING MODELS:")
    print("-" * 40)
    
    # R-squared (higher is better)
    best_r2 = df['R_squared'].idxmax()
    print(f"Best R-squared: {best_r2} ({df.loc[best_r2, 'R_squared']:.4f})")
    
    # MSE (lower is better)
    best_mse = df['MSE'].idxmin()
    print(f"Best MSE: {best_mse} ({df.loc[best_mse, 'MSE']:.4f})")
    
    # RMSE (lower is better)
    best_rmse = df['RMSE'].idxmin()
    print(f"Best RMSE: {best_rmse} ({df.loc[best_rmse, 'RMSE']:.4f})")
    
    # MAE (lower is better)
    best_mae = df['MAE'].idxmin()
    print(f"Best MAE: {best_mae} ({df.loc[best_mae, 'MAE']:.4f})")
    
    # MAPE (lower is better, but exclude inf values)
    valid_mape = df[df['MAPE'] != float('inf')]['MAPE']
    if len(valid_mape) > 0:
        best_mape = valid_mape.idxmin()
        print(f"Best MAPE: {best_mape} ({df.loc[best_mape, 'MAPE']:.2f}%)")
    else:
        print("Best MAPE: All models have infinite MAPE")
    
    print("\n📊 PERFORMANCE IMPROVEMENTS:")
    print("-" * 40)
    
    # Compare with baseline (Linear Regression)
    baseline_r2 = df.loc['Linear Regression', 'R_squared']
    print(f"Linear Regression (Baseline) R-squared: {baseline_r2:.4f}")
    
    for model in df.index:
        if model != 'Linear Regression':
            r2_improvement = df.loc[model, 'R_squared'] - baseline_r2
            if r2_improvement > 0:
                print(f"{model} R-squared improvement: +{r2_improvement:.4f}")
            else:
                print(f"{model} R-squared change: {r2_improvement:.4f}")
    
    print("\n🎯 RECOMMENDATIONS:")
    print("-" * 40)
    
    # Overall best model
    overall_best = df['R_squared'].idxmax()
    print(f"Overall Best Model: {overall_best}")
    print(f"  - R-squared: {df.loc[overall_best, 'R_squared']:.4f}")
    print(f"  - MSE: {df.loc[overall_best, 'MSE']:.4f}")
    print(f"  - MAE: {df.loc[overall_best, 'MAE']:.4f}")
    
    # Model characteristics
    print(f"\nModel Characteristics:")
    if 'Ridge' in overall_best:
        print("  - Uses L2 regularization to prevent overfitting")
        print("  - Good for high-dimensional data with multicollinearity")
    elif 'Polynomial' in overall_best:
        print("  - Captures non-linear relationships")
        print("  - Uses interaction terms and polynomial features")
        print("  - More complex but potentially more accurate")
    
    # Performance interpretation
    best_r2 = df.loc[overall_best, 'R_squared']
    if best_r2 > 0.7:
        print(f"\nPerformance Rating: Excellent (R² = {best_r2:.4f})")
    elif best_r2 > 0.6:
        print(f"\nPerformance Rating: Good (R² = {best_r2:.4f})")
    elif best_r2 > 0.5:
        print(f"\nPerformance Rating: Fair (R² = {best_r2:.4f})")
    else:
        print(f"\nPerformance Rating: Poor (R² = {best_r2:.4f})")


def main():
    """Main function to compare all models."""
    
    # Define paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    results_dir = os.path.join(project_root, 'results')
    
    print("=" * 80)
    print("LIFE EXPECTANCY PREDICTION - MODEL COMPARISON")
    print("=" * 80)
    
    # Define result files
    result_files = {
        'Linear Regression': 'linear_model_results.txt',
        'Ridge Regression': 'ridge_model_results.txt',
        'Polynomial Regression': 'polynomial_model_results.txt',
        'Linear Engineered': 'linear_engineered_model_results.txt'
    }
    
    # Read results from all files
    results = {}
    for model_name, filename in result_files.items():
        file_path = os.path.join(results_dir, filename)
        print(f"Reading results for {model_name}...")
        results[model_name] = read_results_file(file_path)
    
    # Create comparison table
    print("\nCreating comparison table...")
    df = create_comparison_table(results)
    
    # Display results
    print("\n" + "=" * 80)
    print("MODEL COMPARISON RESULTS")
    print("=" * 80)
    print(df.to_string())
    
    # Print detailed analysis
    print_analysis(df)
    
    # Save comparison to file
    comparison_file = os.path.join(results_dir, 'model_comparison.txt')
    with open(comparison_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("LIFE EXPECTANCY PREDICTION - MODEL COMPARISON\n")
        f.write("=" * 80 + "\n\n")
        f.write("COMPARISON TABLE:\n")
        f.write("-" * 40 + "\n")
        f.write(df.to_string() + "\n\n")
        
        # Add analysis
        f.write("ANALYSIS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Best R-squared: {df['R_squared'].idxmax()} ({df['R_squared'].max():.4f})\n")
        f.write(f"Best MSE: {df['MSE'].idxmin()} ({df['MSE'].min():.4f})\n")
        f.write(f"Best RMSE: {df['RMSE'].idxmin()} ({df['RMSE'].min():.4f})\n")
        f.write(f"Best MAE: {df['MAE'].idxmin()} ({df['MAE'].min():.4f})\n")
        
        # R-squared improvements
        baseline_r2 = df.loc['Linear Regression', 'R_squared']
        f.write(f"\nR-squared Improvements over Linear Regression:\n")
        for model in df.index:
            if model != 'Linear Regression':
                improvement = df.loc[model, 'R_squared'] - baseline_r2
                f.write(f"  {model}: {improvement:+.4f}\n")
    
    print(f"\nComparison saved to: {comparison_file}")
    print("\n" + "=" * 80)
    print("MODEL COMPARISON COMPLETED!")
    print("=" * 80)


if __name__ == "__main__":
    main()
