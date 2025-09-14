import argparse
import os
import numpy as np
import pandas as pd
import pickle
import sys
import random
from typing import Tuple

# Add the current directory to Python path to import the LinearRegressionEngineered class
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_linear_engineered_2 import LinearRegressionEngineered


def load_model(model_path: str):
    """
    Load the trained model from pickle file.
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        Loaded LinearRegressionEngineered model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    return model


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the same data preprocessing transformations as in data_preprocessing.py
    
    Args:
        df: Raw DataFrame
        
    Returns:
        Preprocessed DataFrame
    """
    print("Applying life expectancy data preprocessing transformations...")
    
    # Step 1: Clean Life expectancy (drop entries with missing life expectancy)
    initial_count = len(df)
    df = df.dropna(subset=['Life expectancy '])
    final_count = len(df)
    removed_count = initial_count - final_count
    print(f"Removed {removed_count} entries with missing Life expectancy")
    
    # Step 2: Encode Status feature (before dropping Country)
    status_mapping = {'Developed': 0.5, 'Developing': -0.5}
    df['Status'] = df['Status'].map(status_mapping)
    print("Encoded Status feature: Developed=0.5, Developing=-0.5")
    
    # Step 3: Handle missing values for Adult Mortality (needs Country column)
    missing_count = df['Adult Mortality'].isna().sum()
    if missing_count > 0:
        df_sorted = df.sort_values(['Country', 'Year']).copy()
        for idx in df_sorted[df_sorted['Adult Mortality'].isna()].index:
            country = df_sorted.loc[idx, 'Country']
            year = df_sorted.loc[idx, 'Year']
            country_data = df_sorted[
                (df_sorted['Country'] == country) & 
                (df_sorted['Year'] < year) & 
                (df_sorted['Adult Mortality'].notna())
            ]
            if len(country_data) > 0:
                median_value = country_data['Adult Mortality'].median()
                df_sorted.loc[idx, 'Adult Mortality'] = median_value
            else:
                overall_median = df['Adult Mortality'].median()
                df_sorted.loc[idx, 'Adult Mortality'] = overall_median
        df = df_sorted.sort_index()
    print(f"Handled {missing_count} missing Adult Mortality values")
    
    # Step 4: Handle missing values for GDP (needs Country column)
    missing_count = df['GDP'].isna().sum()
    if missing_count > 0:
        df_sorted = df.sort_values(['Country', 'Year']).copy()
        for idx in df_sorted[df_sorted['GDP'].isna()].index:
            country = df_sorted.loc[idx, 'Country']
            year = df_sorted.loc[idx, 'Year']
            prev_year_data = df_sorted[
                (df_sorted['Country'] == country) & 
                (df_sorted['Year'] == year - 1) & 
                (df_sorted['GDP'].notna())
            ]
            if len(prev_year_data) > 0:
                prev_gdp = prev_year_data['GDP'].iloc[0]
                df_sorted.loc[idx, 'GDP'] = prev_gdp
            else:
                overall_mean = df['GDP'].mean()
                df_sorted.loc[idx, 'GDP'] = overall_mean
        df = df_sorted.sort_index()
    print(f"Handled {missing_count} missing GDP values")
    
    # Step 5: One-hot encode Country (after all country-specific operations)
    country_dummies = pd.get_dummies(df['Country'], prefix='Country')
    df = df.drop('Country', axis=1)
    df = pd.concat([df, country_dummies], axis=1)
    print(f"One-hot encoded Country feature: {len(country_dummies.columns)} countries")
    
    # Step 6: Handle missing values for Alcohol
    missing_count = df['Alcohol'].isna().sum()
    if missing_count > 0:
        median_value = df['Alcohol'].median()
        df['Alcohol'] = df['Alcohol'].fillna(median_value)
    print(f"Handled {missing_count} missing Alcohol values")
    
    # Step 7: Drop Hepatitis B and Measles features
    features_to_drop = ['Hepatitis B', 'Measles', 'Population']
    for feature in features_to_drop:
        if feature in df.columns:
            df = df.drop(feature, axis=1)
    print(f"Dropped features: {features_to_drop}")
    
    # Step 8: Handle BMI (drop rows with missing BMI)
    initial_count = len(df)
    df = df.dropna(subset=[' BMI '])
    final_count = len(df)
    removed_count = initial_count - final_count
    print(f"Removed {removed_count} rows with missing BMI")
    
    # Step 9: Handle missing values for Polio
    missing_count = df['Polio'].isna().sum()
    if missing_count > 0:
        median_value = df['Polio'].median()
        df['Polio'] = df['Polio'].fillna(median_value)
    print(f"Handled {missing_count} missing Polio values")
    
    # Step 10: Handle missing values for Total expenditure
    missing_count = df['Total expenditure'].isna().sum()
    if missing_count > 0:
        mean_value = df['Total expenditure'].mean()
        df['Total expenditure'] = df['Total expenditure'].fillna(mean_value)
    print(f"Handled {missing_count} missing Total expenditure values")
    
    # Step 11: Handle missing values for Diphtheria
    missing_count = df['Diphtheria '].isna().sum()
    if missing_count > 0:
        mean_value = df['Diphtheria '].mean()
        df['Diphtheria '] = df['Diphtheria '].fillna(mean_value)
    print(f"Handled {missing_count} missing Diphtheria values")
    
    # Step 12: Handle thinness features
    for feature in [' thinness  1-19 years', ' thinness 5-9 years']:
        if feature in df.columns:
            missing_count = df[feature].isna().sum()
            if missing_count > 0:
                median_value = df[feature].median()
                df[feature] = df[feature].fillna(median_value)
            print(f"Handled {missing_count} missing {feature} values")
    
    # Step 13: Handle Income composition (drop rows with missing values)
    if 'Income composition of resources' in df.columns:
        initial_count = len(df)
        df = df.dropna(subset=['Income composition of resources'])
        final_count = len(df)
        removed_count = initial_count - final_count
        print(f"Removed {removed_count} rows with missing Income composition")
    
    # Step 14: Handle missing values for Schooling
    missing_count = df['Schooling'].isna().sum()
    if missing_count > 0:
        mean_value = df['Schooling'].mean()
        df['Schooling'] = df['Schooling'].fillna(mean_value)
    print(f"Handled {missing_count} missing Schooling values")
    
    # Step 15: Normalize all features to [-1, 1] range
    target_column = 'Life expectancy '
    feature_columns = [col for col in df.columns if col != target_column]
    print(f"Normalizing {len(feature_columns)} features to [-1, 1] range")
    
    for column in feature_columns:
        if df[column].dtype in ['int64', 'float64']:
            min_val = df[column].min()
            max_val = df[column].max()
            if max_val != min_val:
                df[column] = 2 * (df[column] - min_val) / (max_val - min_val) - 1
    
    print("Data preprocessing complete.")
    return df


def load_and_preprocess_data(data_path: str) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Load and preprocess the data using the same transformations as training.
    
    Args:
        data_path: Path to the CSV file
        
    Returns:
        Tuple of (features, target, feature_names) - target may be None for test data
    """
    # Load raw data
    df = pd.read_csv(data_path)
    print(f"Loaded data with shape: {df.shape}")
    
    # Apply preprocessing
    df_processed = preprocess_data(df.copy())
    
    # Check if target column exists
    has_target = 'Life expectancy ' in df_processed.columns
    
    if has_target:
        # Training data with target
        target = df_processed['Life expectancy '].values
        features = df_processed.drop(columns=['Life expectancy ']).values
        feature_names = df_processed.drop(columns=['Life expectancy ']).columns.tolist()
    else:
        # Test data without target
        target = None
        features = df_processed.values
        feature_names = df_processed.columns.tolist()
    
    print(f"Processed data shape: {features.shape}")
    if has_target:
        print(f"Target shape: {target.shape}")
    
    return features, target, feature_names


def clean_data(X: np.ndarray, y: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Clean the data by handling NaN and infinite values.
    
    Args:
        X: Feature matrix
        y: Target values (optional)
        
    Returns:
        Tuple of (cleaned_features, cleaned_target)
    """
    # Check for NaN or infinite values in input data
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    
    if y is not None:
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            y = np.nan_to_num(y, nan=np.mean(y), posinf=np.max(y), neginf=np.min(y))
    
    return X, y


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculate regression evaluation metrics.
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        
    Returns:
        Dictionary containing evaluation metrics
    """
    # Mean Squared Error
    mse = np.mean((y_true - y_pred) ** 2)
    
    # Root Mean Squared Error
    rmse = np.sqrt(mse)
    
    # R-squared
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'R_squared': r_squared
    }


def save_metrics(metrics: dict, output_path: str):
    """
    Save metrics to file in standardized format.
    
    Args:
        metrics: Dictionary containing evaluation metrics
        output_path: Path where metrics will be saved
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("Regression Metrics:\n")
        f.write(f"Mean Squared Error (MSE): {metrics['MSE']:.2f}\n")
        f.write(f"Root Mean Squared Error (RMSE): {metrics['RMSE']:.2f}\n")
        f.write(f"R-squared (R²) Score: {metrics['R_squared']:.2f}\n")


def save_predictions(predictions: np.ndarray, output_path: str):
    """
    Save predictions to CSV file in standardized format.
    
    Args:
        predictions: Array of predictions
        output_path: Path where predictions will be saved
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save as single column CSV without header
    df = pd.DataFrame(predictions)
    df.to_csv(output_path, index=False, header=False)


def main():
    """Main function for model evaluation."""
    parser = argparse.ArgumentParser(description='Evaluate life expectancy regression model')
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the saved model file')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to the data CSV file that includes features and true labels')
    parser.add_argument('--metrics_output_path', type=str, required=True,
                       help='Path where evaluation metrics will be saved')
    parser.add_argument('--predictions_output_path', type=str, required=True,
                       help='Path where predictions will be saved')
    
    args = parser.parse_args()
    
    try:
        # Load model
        print(f"Loading model from: {args.model_path}")
        model = load_model(args.model_path)
        
        # Load and preprocess data
        print(f"Loading and preprocessing data from: {args.data_path}")
        X, y, feature_names = load_and_preprocess_data(args.data_path)
        
        # Clean data
        X_clean, y_clean = clean_data(X, y)
        
        # Make predictions
        print("Generating predictions...")
        y_pred = model.predict(X_clean)
        
        # Calculate and save metrics (only if target exists)
        if y_clean is not None:
            metrics = calculate_metrics(y_clean, y_pred)
            save_metrics(metrics, args.metrics_output_path)
            print(f"Metrics saved to: {args.metrics_output_path}")
        else:
            print("No target values found - skipping metrics calculation")
        
        # Save predictions
        save_predictions(y_pred, args.predictions_output_path)
        print(f"Predictions saved to: {args.predictions_output_path}")
        
        print(f"Evaluation completed successfully!")
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()