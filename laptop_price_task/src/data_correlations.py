import pandas as pd
import os

def analyze_laptop_price_correlations():
    """
    Loads the fully preprocessed and normalized laptop data, calculates the
    Pearson correlation of all features with the target variable 'Price',
    and saves the results to a CSV file.
    """
    try:
        # --- 1. DEFINE FILE PATHS ---
        base_dir = os.path.join(os.path.dirname(__file__), '..')
        input_path = os.path.join(base_dir, 'data', 'train_data_transformed.csv')
        output_path = os.path.join(base_dir, 'results', 'feature_price_correlations.csv')

        # Create the results directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # --- 2. LOAD THE TRANSFORMED DATA ---
        print(f"Loading transformed data from: {input_path}")
        df = pd.read_csv(input_path)

        # --- 3. CALCULATE PEARSON CORRELATIONS ---
        print("Calculating Pearson correlation of all features with 'Price'...")
        
        # The .corr() method automatically handles the boolean/flag columns by treating them as 0 and 1
        correlation_matrix = df.corr(method='pearson')
        
        # Isolate the correlations with the 'Price' column and sort them
        price_correlations = correlation_matrix['Price'].sort_values(ascending=False)

        # --- 4. SAVE THE RESULTS TO A CSV FILE ---
        # Convert the Series to a DataFrame for saving
        correlation_df = price_correlations.reset_index()
        correlation_df.columns = ['Feature', 'Pearson_Correlation_with_Price']
        
        correlation_df.to_csv(output_path, index=False)
        
        print(f"\n--- Analysis Complete ---")
        print(f"Correlation results saved to: {output_path}")

        # Display the top 10 most correlated features
        print("\nTop 10 features most positively correlated with Price:")
        print(correlation_df.head(11)) # .head(11) to include 'Price' itself

    except FileNotFoundError:
        print(f"Error: The file '{os.path.basename(input_path)}' was not found.")
        print("Please run the data_preprocessing.py script first to generate the transformed data.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    analyze_laptop_price_correlations()