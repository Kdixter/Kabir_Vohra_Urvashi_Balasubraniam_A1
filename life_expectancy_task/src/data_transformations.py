import pandas as pd
import numpy as np
import os

def preprocess_life_expectancy_data():
    """
    Loads and preprocesses the life expectancy dataset according to a detailed set of rules.

    This function performs the following steps:
    1.  Loads the data.
    2.  Handles missing values in the target and key predictors by dropping rows.
    3.  Imputes missing values for various features using mean or median strategies.
    4.  Performs a grouped imputation for 'Adult Mortality' based on the country's average.
    5.  Encodes the 'Status' feature numerically.
    6.  Drops the 'Population' column.
    7.  Returns a cleaned DataFrame ready for model training.

    Returns:
        pandas.DataFrame: The preprocessed and cleaned dataset, or None if loading fails.
    """
    try:
        # --- 1. LOAD THE DATASET ---
        # Assumes the script is run from a location where this relative path is valid.
        file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'train_data.csv')
        df = pd.read_csv(file_path)
        print(f"Original dataset shape: {df.shape}")

        # --- 2. APPLY DATA TRANSFORMATIONS ---

        # Rule: Drop entries without a "Life expectancy " value
        df.dropna(subset=['Life expectancy '], inplace=True)

        # Rule: For "Income composition of resources" and "Schooling", drop entries if NULL
        df.dropna(subset=['Income composition of resources', 'Schooling'], inplace=True)
        print(f"Shape after dropping key NULL rows: {df.shape}")

        # Rule: Fill "Adult Mortality" NULLs with the mean of the same country from other years
        # We use transform to broadcast the country's mean mortality back to each row.
        df['Adult Mortality'] = df['Adult Mortality'].fillna(df.groupby('Country')['Adult Mortality'].transform('mean'))
        # If a country has only NaN values, its mean will also be NaN. Fill any remaining NaNs with the global mean.
        df['Adult Mortality'].fillna(df['Adult Mortality'].mean(), inplace=True)

        # Rule: Change "Status" to 1 if Developed and 0.5 if Developing
        df['Status'] = df['Status'].map({'Developed': 1.0, 'Developing': 0.5})

        # Rule: Drop the "Population" column
        df.drop(columns=['Population'], inplace=True)

        # --- 3. IMPUTE MISSING VALUES (NULL/NaN) ---

        # Define columns for mean and median imputation
        cols_for_mean_imputation = [
            'infant deaths', 'percentage expenditure', 'Hepatitis B', ' BMI ',
            ' thinness  1-19 years', ' thinness 5-9 years'
        ]

        cols_for_median_imputation = [
            'Alcohol', 'Measles ', 'under-five deaths ', 'Polio',
            'Total expenditure', 'Diphtheria ', ' HIV/AIDS', 'GDP'
        ]

        # Apply mean imputation
        for col in cols_for_mean_imputation:
            # Note: For 'percentage expenditure', we only fill NaNs, leaving existing 0s untouched.
            mean_val = df[col].mean()
            df[col].fillna(mean_val, inplace=True)

        # Apply median imputation
        for col in cols_for_median_imputation:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)

        print("All specified transformations and imputations are complete.")
        print(f"Final dataset shape after preprocessing: {df.shape}")
        
        # Final check for any remaining nulls
        null_counts = df.isnull().sum()
        if null_counts.sum() == 0:
            print("No missing values remain in the dataset.")
        else:
            print("Warning: Some missing values still remain:")
            print(null_counts[null_counts > 0])
            
        return df

    except FileNotFoundError:
        print(f"Error: The file was not found. Please ensure 'train_data.csv' is in the '../data/' directory relative to this script.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during preprocessing: {e}")
        return None

# This block allows you to run the script directly to test the preprocessing function
# and save the transformed data to a new CSV file.
if __name__ == "__main__":
    print("--- Running Data Preprocessing ---")
    cleaned_data = preprocess_life_expectancy_data()
    
    if cleaned_data is not None:
        # --- SAVE THE TRANSFORMED DATA ---
        try:
            # Define the path for the output file
            output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'train_data_transformed.csv')
            
            # Save the DataFrame to the new CSV file, without the index column
            cleaned_data.to_csv(output_path, index=False)
            
            print(f"\n--- Preprocessing Complete ---")
            print(f"Transformed data successfully saved to: {output_path}")

        except Exception as e:
            print(f"\nAn error occurred while saving the file: {e}")
