import pandas as pd
import numpy as np
import os

def run_full_preprocessing_pipeline():
    """
    Runs a full data preprocessing pipeline on the original dataset.

    This single function encapsulates all cleaning and transformation steps:
    1.  Loads the original 'life_expectancy.csv' data.
    2.  Removes a specified list of irrelevant features.
    3.  Handles missing values by dropping rows and using imputation.
    4.  Applies a custom encoding to the 'Status' feature.
    5.  Normalizes all predictive features to a range of [-1, 1].
    6.  Splits the data into training (90%) and testing (10%) sets.
    7.  Saves the final datasets to 'poly_train.csv' and 'poly_test.csv'.
    """
    try:
        # --- 1. LOAD THE ORIGINAL DATASET ---
        base_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        input_path = os.path.join(base_dir, 'life_expectancy.csv')
        train_output_path = os.path.join(base_dir, 'poly_train.csv')
        test_output_path = os.path.join(base_dir, 'poly_test.csv')
        
        print(f"Loading original dataset from: {input_path}")
        df = pd.read_csv(input_path)
        print(f"Original dataset shape: {df.shape}")

        # --- 2. REMOVE SPECIFIED FEATURES ---
        columns_to_drop = [
            'Total expenditure', 'Hepatitis B', 'Year', 'Population',
            'Measles ', 'infant deaths', 'under-five deaths '
        ]
        df.drop(columns=columns_to_drop, inplace=True)
        print(f"Dropped {len(columns_to_drop)} specified columns. New shape: {df.shape}")

        # --- 3. HANDLE CRITICAL MISSING VALUES BY DROPPING ROWS ---
        df.dropna(subset=['Life expectancy '], inplace=True)
        df.dropna(subset=['Income composition of resources', 'Schooling'], inplace=True)
        print(f"Shape after dropping rows with critical missing values: {df.shape}")

        # --- 4. IMPUTE MISSING VALUES (NULL/NaN) ---
        print("Starting imputation for missing values...")
        
        df['Adult Mortality'] = df.groupby('Country')['Adult Mortality'].transform(lambda x: x.fillna(x.mean()))
        df['Adult Mortality'].fillna(df['Adult Mortality'].mean(), inplace=True)

        cols_for_median = ['Alcohol', 'Polio', 'Diphtheria ', ' HIV/AIDS', 'GDP']
        for col in cols_for_median:
            df[col].fillna(df[col].median(), inplace=True)

        cols_for_mean = [
            'percentage expenditure', ' BMI ', ' thinness  1-19 years',
            ' thinness 5-9 years'
        ]
        for col in cols_for_mean:
            df[col].fillna(df[col].mean(), inplace=True)
        
        print("Imputation complete.")

        # --- 5. ENCODE CATEGORICAL FEATURES (UPDATED MAPPING) ---
        df['Status'] = df['Status'].map({'Developed': 0.5, 'Developing': -0.5})
        print("Encoded 'Status' feature numerically (Developed: 0.5, Developing: -0.5).")

        # --- 6. NORMALIZE FEATURES TO [-1, 1] RANGE ---
        print("Normalizing features to the range [-1, 1]...")
        
        features_to_scale = df.drop(columns=['Country', 'Life expectancy '])
        non_features = df[['Country', 'Life expectancy ']]

        status_col = features_to_scale[['Status']]
        other_features = features_to_scale.drop(columns=['Status'])

        scaled_other_features = 2 * (other_features - other_features.min()) / (other_features.max() - other_features.min()) - 1
        
        final_df = pd.concat([non_features, status_col, scaled_other_features], axis=1)
        print("Feature normalization complete.")

        # --- 7. SPLIT DATA INTO TRAINING AND TESTING SETS ---
        print("\nSplitting data into training (90%) and testing (10%) sets...")

        # Shuffle the DataFrame to ensure the split is random and reproducible
        df_shuffled = final_df.sample(frac=1, random_state=42)

        # Calculate the split index
        split_index = int(0.9 * len(df_shuffled))

        # Create the training and testing sets
        train_df = df_shuffled[:split_index]
        test_df = df_shuffled[split_index:]

        # --- 8. SAVE THE FINAL DATASETS ---
        train_df.to_csv(train_output_path, index=False)
        test_df.to_csv(test_output_path, index=False)
        
        print(f"\n--- Data Pipeline Complete ---")
        print(f"Training data saved to: {train_output_path} ({len(train_df)} rows)")
        print(f"Testing data saved to: {test_output_path} ({len(test_df)} rows)")

    except FileNotFoundError:
        print(f"Error: The original data file was not found at {input_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    run_full_preprocessing_pipeline()

