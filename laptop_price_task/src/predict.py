import argparse
import os
import numpy as np
import pandas as pd
import pickle
import sys
import re
from typing import Tuple

# Add the current directory to Python path to import the LinearRegression class
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_model import LinearRegression


def load_model(model_path: str): # takes in the model path and gives out the model
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    return model

# Apply the same data preprocessing transformations as I have done on my end:
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    
    print("Applying data preprocessing transformations...")
    
    # cleaning:
    df['Ram'] = df['Ram'].str.replace('GB', '').astype(int)
    df['Weight'] = df['Weight'].str.replace('kg', '').astype(float)
    
    # feature enginerring:
    # ScreenResolution
    df['Touchscreen'] = df['ScreenResolution'].apply(lambda x: 1 if 'Touchscreen' in x else 0)
    df['IPS_Panel'] = df['ScreenResolution'].apply(lambda x: 1 if 'IPS' in x else 0)
    df['X_res'] = df['ScreenResolution'].str.split('x').str[0].str.findall(r'(\d+)').str[-1].astype(int)
    df['Y_res'] = df['ScreenResolution'].str.split('x').str[1].astype(int)
    df.drop(columns=['ScreenResolution'], inplace=True)

    # 0 valuen handling:
    for col in ['X_res', 'Y_res']:
        df[col] = df[col].replace(0, np.nan)
        df[col].fillna(df[col].median(), inplace=True)
        
    # Cpu
    df['Cpu_Brand'] = df['Cpu'].apply(lambda x: " ".join(x.split()[0:3]))
    df['Cpu_Ghz'] = df['Cpu'].apply(lambda x: float(x.split()[-1].replace('GHz', '')))
    df.drop(columns=['Cpu'], inplace=True)

    # Memory
    df['Memory'] = df['Memory'].astype(str).replace(r'\.0', '', regex=True)
    df["Memory"] = df["Memory"].str.replace('GB', '')
    df["Memory"] = df["Memory"].str.replace('TB', '000')
    new = df["Memory"].str.split("+", n=1, expand=True)
    df["first"] = new[0].str.strip()
    df["second"] = new[1]
    df["SSD"] = df["first"].apply(lambda x: int(re.search(r'\d+', x).group()) if "SSD" in x else 0)
    df["HDD"] = df["first"].apply(lambda x: int(re.search(r'\d+', x).group()) if "HDD" in x else 0)
    df["second"] = df["second"].fillna("0")
    df["SSD"] += df["second"].apply(lambda x: int(re.search(r'\d+', x).group()) if "SSD" in x else 0)
    df["HDD"] += df["second"].apply(lambda x: int(re.search(r'\d+', x).group()) if "HDD" in x else 0)
    df.drop(columns=['Memory', 'first', 'second'], inplace=True)

    # Gpu
    df['Gpu_Brand'] = df['Gpu'].apply(lambda x: x.split()[0])
    df = df[df['Gpu_Brand'] != 'ARM']
    df.drop(columns=['Gpu'], inplace=True)

    # creating interaction features
    df['Total_Pixels'] = df['X_res'] * df['Y_res']
    df['Ram_SSD_Interaction'] = df['Ram'] * df['SSD']

    # creating one hot encoding:
    df = pd.get_dummies(df, columns=['Company', 'TypeName', 'OpSys', 'Cpu_Brand', 'Gpu_Brand'], drop_first=True)

    #  normalisation and checking if "Price" exists
    has_price = 'Price' in df.columns
    if has_price:
        price_col = df['Price']
        df_features = df.drop(columns=['Price'])
    else:
        df_features = df.copy()
    
    cols_to_scale = [col for col in df_features.columns if df_features[col].dtype != 'bool' and len(df_features[col].unique()) > 2]
    df_to_scale = df_features[cols_to_scale]
    df_flags = df_features.drop(columns=cols_to_scale)
    
    df_scaled_continuous = (df_to_scale - df_to_scale.min()) / (df_to_scale.max() - df_to_scale.min())
    
    if has_price:
        df_final = pd.concat([price_col, df_scaled_continuous, df_flags], axis=1)
    else:
        df_final = pd.concat([df_scaled_continuous, df_flags], axis=1)
    
    print("Data preprocessing complete.")
    return df_final


def load_and_preprocess_data(data_path: str) -> Tuple[np.ndarray, np.ndarray]:
    
    # Load raw data
    df = pd.read_csv(data_path)
    print(f"Loaded data with shape: {df.shape}")
    
    # Pre-process it:
    df_processed = preprocess_data(df.copy())
    
    # Check if target column exists (if not then something ahs gone wrong):
    has_target = 'Price' in df_processed.columns
    
    if has_target:
        # Training data with target
        target = df_processed['Price'].values
        features = df_processed.drop(columns=['Price']).values
    else:
        # Test data without target
        target = None
        features = df_processed.values
    
    print(f"Processed data shape: {features.shape}")
    if has_target:
        print(f"Target shape: {target.shape}")
    
    return features, target


def clean_data(X: np.ndarray, y: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    
    # Check for NaN or infinite values in input data
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    
    if y is not None:
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            y = np.nan_to_num(y, nan=np.mean(y), posinf=np.max(y), neginf=np.min(y))
    
    return X, y


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    
    # Mean Squared Error (mse)
    mse = np.mean((y_true - y_pred) ** 2)
    
    # root(mse)
    rmse = np.sqrt(mse)
    
    # R^2
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'R_squared': r_squared
    }


def save_metrics(metrics: dict, output_path: str): # self explanatory
    
    # Create output directory if it doesn't exist (it exists)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("Regression Metrics:\n")
        f.write(f"Mean Squared Error (MSE): {metrics['MSE']:.2f}\n")
        f.write(f"Root Mean Squared Error (RMSE): {metrics['RMSE']:.2f}\n")
        f.write(f"R-squared (R²) Score: {metrics['R_squared']:.2f}\n")


# saves all predictions in a seperate csv that will be created when
# this file is run
def save_predictions(predictions: np.ndarray, output_path: str):
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save as single column CSV without header
    df = pd.DataFrame(predictions)
    df.to_csv(output_path, index=False, header=False)

# main function that runs all above in the order that they are supposed to be
def main():
    """Main function for model evaluation."""
    parser = argparse.ArgumentParser(description='Evaluate regression model')
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the saved model file')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to the data CSV file that includes features and true labels')
    parser.add_argument('--metrics_output_path', type=str, required=True,
                       help='Path where evaluation metrics will be saved')
    parser.add_argument('--predictions_output_path', type=str, required=True,
                       help='Path where predictions will be saved')
    
    args = parser.parse_args()
    
    try: # for error handling
        # Load model
        print(f"Loading model from: {args.model_path}")
        model = load_model(args.model_path)
        
        # Load and preprocess data
        print(f"Loading and preprocessing data from: {args.data_path}")
        X, y = load_and_preprocess_data(args.data_path)
        
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