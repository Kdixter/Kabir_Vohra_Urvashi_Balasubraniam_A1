import pandas as pd
import numpy as np
import os

def save_life_expectancy_correlations():
    """
    Loads, preprocesses, and calculates the Pearson correlation of features
    with life expectancy, saving the result to a CSV file.
    """
    try:
        # --- 1. DEFINE FILE PATHS ---
        # Assumes the script is run from the project's root directory.
        data_path = os.path.join('life_expectancy_task', 'data', 'train_data.csv')
        output_path = os.path.join('life_expectancy_task', 'results', 'feature_correlations.csv')

        # Create the results directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # --- 2. LOAD AND PREPROCESS THE DATA ---
        data = pd.read_csv(data_path)

        # Drop rows where the target feature 'Life expectancy ' is missing
        data.dropna(subset=['Life expectancy '], inplace=True)

        # Convert 'Status' column to numerical (1 for Developed, 0 for Developing)
        data['Status'] = data['Status'].apply(lambda x: 1 if x == 'Developed' else 0)

        # Handle missing values (NaNs) in all numeric columns by imputing with the median
        numeric_cols = data.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            if data[col].isnull().sum() > 0:
                median_val = data[col].median()
                data[col].fillna(median_val, inplace=True)
        
        # Drop the non-numeric 'Country' column before correlation
        data_numeric = data.drop('Country', axis=1)
        
        # --- 3. CALCULATE CORRELATIONS ---
        correlation_matrix = data_numeric.corr(method='pearson')
        target_correlation = correlation_matrix['Life expectancy '].sort_values(ascending=False)

        # --- 4. SAVE THE RESULTS TO A CSV FILE ---
        
        # Convert the Series to a DataFrame for saving
        correlation_df = target_correlation.reset_index()
        correlation_df.columns = ['Feature', 'Pearson_Correlation']
        
        # Save the DataFrame to the specified output path
        correlation_df.to_csv(output_path, index=False)
        
        print(f"Correlation data successfully saved to: {output_path}")

    except FileNotFoundError:
        print(f"Error: The file was not found at {data_path}")
        print("Please ensure 'train_data.csv' is in the 'life_expectancy_task/data/' directory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Run the analysis and save the file
if __name__ == "__main__":
    save_life_expectancy_correlations()