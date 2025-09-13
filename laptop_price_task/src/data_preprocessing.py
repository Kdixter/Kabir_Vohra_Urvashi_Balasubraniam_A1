import pandas as pd
import numpy as np
import os
import re

def run_full_pipeline_and_analyze():
    """
    Consolidated script that performs the entire data preprocessing pipeline
    for the laptop dataset and concludes with a correlation analysis.
    
    Steps:
    1. Loads the raw data.
    2. Cleans and engineers basic features.
    3. Creates advanced interaction features based on analysis.
    4. Normalizes all numerical features.
    5. Saves the final, cleaned dataset.
    6. Calculates and saves the feature correlation with Price.
    """
    try:
        # --- 1. DEFINE FILE PATHS ---
        base_dir = os.path.join(os.path.dirname(__file__), '..')
        input_path = os.path.join(base_dir, 'data', 'train_data.csv')
        # The final, fully processed data for model training
        final_data_output_path = os.path.join(base_dir, 'data', 'train_data_final_processed.csv')
        # The correlation analysis results
        correlation_output_path = os.path.join(base_dir, 'results', 'feature_price_correlations.csv')
        os.makedirs(os.path.join(base_dir, 'results'), exist_ok=True)

        # --- 2. LOAD AND CLEAN DATA ---
        df = pd.read_csv(input_path)
        print("Original data loaded successfully.")
        df['Ram'] = df['Ram'].str.replace('GB', '').astype(int)
        df['Weight'] = df['Weight'].str.replace('kg', '').astype(float)
        
        # --- 3. ADVANCED FEATURE ENGINEERING ---
        # ScreenResolution
        df['Touchscreen'] = df['ScreenResolution'].apply(lambda x: 1 if 'Touchscreen' in x else 0)
        df['IPS_Panel'] = df['ScreenResolution'].apply(lambda x: 1 if 'IPS' in x else 0)
        df['X_res'] = df['ScreenResolution'].str.split('x').str[0].str.findall(r'(\d+)').str[-1].astype(int)
        df['Y_res'] = df['ScreenResolution'].str.split('x').str[1].astype(int)
        df.drop(columns=['ScreenResolution'], inplace=True)

        # Impute invalid zero values in resolution
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
        print("Advanced feature engineering complete.")

        # --- 4. NEW INTERACTION FEATURES ---
        print("Creating new interaction features based on correlation analysis...")
        # A feature for total pixels
        df['Total_Pixels'] = df['X_res'] * df['Y_res']
        # A feature combining Ram and SSD performance
        df['Ram_SSD_Interaction'] = df['Ram'] * df['SSD']
        print(" - Created 'Total_Pixels' and 'Ram_SSD_Interaction' features.")

        # --- 5. ONE-HOT ENCODING ---
        df = pd.get_dummies(df, columns=['Company', 'TypeName', 'OpSys', 'Cpu_Brand', 'Gpu_Brand'], drop_first=True)
        print("Applied One-Hot Encoding.")

        # --- 6. NORMALIZE NUMERICAL DATA ---
        print("Normalizing numerical features...")
        price_col = df['Price']
        df_features = df.drop(columns=['Price'])
        
        cols_to_scale = [col for col in df_features.columns if df_features[col].dtype != 'bool' and len(df_features[col].unique()) > 2]
        df_to_scale = df_features[cols_to_scale]
        df_flags = df_features.drop(columns=cols_to_scale)
        
        df_scaled_continuous = (df_to_scale - df_to_scale.min()) / (df_to_scale.max() - df_to_scale.min())
        
        df_final = pd.concat([price_col, df_scaled_continuous, df_flags], axis=1)
        print("Normalization complete.")

        # --- 7. SAVE THE FINAL PROCESSED DATA ---
        df_final.to_csv(final_data_output_path, index=False)
        print(f"\nFinal processed data for training saved to: {final_data_output_path}")

        # --- 8. CALCULATE AND SAVE CORRELATIONS ---
        print("Calculating Pearson correlation of final features with 'Price'...")
        correlation_matrix = df_final.corr(method='pearson')
        price_correlations = correlation_matrix['Price'].sort_values(ascending=False)
        
        correlation_df = price_correlations.reset_index()
        correlation_df.columns = ['Feature', 'Pearson_Correlation_with_Price']
        correlation_df.to_csv(correlation_output_path, index=False)
        
        print(f"Correlation results saved to: {correlation_output_path}")
        print("\nTop 10 features most positively correlated with Price:")
        print(correlation_df.head(11))

    except FileNotFoundError:
        print(f"Error: The file 'train_data.csv' was not found in the 'data' directory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    run_full_pipeline_and_analyze()