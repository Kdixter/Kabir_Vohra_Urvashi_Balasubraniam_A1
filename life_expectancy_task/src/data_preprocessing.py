import pandas as pd
import os

def load_life_expectancy_data():
    """
    Loads the life expectancy dataset.

    Returns:
        pandas.DataFrame: A DataFrame containing the loaded dataset,
                          or None if the file is not found.
    """
    try:
        # Construct a relative path to the data file
        # Goes up one level from 'src' and then into 'data'
        file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'train_data.csv')
        
        # Load the CSV file into a pandas DataFrame
        data = pd.read_csv(file_path)
        
        print("Data loaded successfully!")
        print("Shape of the dataset:", data.shape)
        print("\nFirst 5 rows of the dataset:")
        print(data.head())
        
        return data

    except FileNotFoundError:
        print(f"Error: The file was not found at the expected location.")
        print("Please ensure that 'train_data.csv' exists in the '../data/' directory.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

# This block allows the script to be run directly to test the data loading
if __name__ == "__main__":
    print("Attempting to load dataset...")
    life_expectancy_df = load_life_expectancy_data()

    if life_expectancy_df is not None:
        print("\n--- Data Loading Test Complete ---")
        # You can now proceed with data preprocessing and model training
        # using the 'life_expectancy_df' DataFrame.