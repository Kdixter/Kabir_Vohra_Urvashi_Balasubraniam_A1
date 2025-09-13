import pandas as pd
import os

def split_final_data(test_split_ratio=0.1, random_state=42):
    """
    Loads the final, fully processed dataset, splits it into training and
    testing sets, and saves them to their final destination files.

    Args:
        test_split_ratio (float): The proportion of the dataset for the test set.
        random_state (int): Seed for the random shuffle to ensure reproducibility.
    """
    try:
        # --- 1. DEFINE FILE PATHS ---
        base_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        # This is the input file created by your all-in-one processing script
        input_path = os.path.join(base_dir, 'train_data_final_processed.csv')
        
        # Define the output paths for the final split files
        train_output_path = os.path.join(base_dir, 'final_train_data_processed.csv')
        test_output_path = os.path.join(base_dir, 'final_test_data.csv')

        # --- 2. LOAD THE PROCESSED DATA ---
        print(f"Loading fully processed data from: {input_path}")
        df = pd.read_csv(input_path)

        # --- 3. SHUFFLE AND SPLIT THE DATA ---
        print(f"Splitting data into {100 - test_split_ratio*100}% training and {test_split_ratio*100}% testing sets...")

        # Shuffle the DataFrame to ensure a random split
        df_shuffled = df.sample(frac=1, random_state=random_state)

        # Calculate the split index
        split_index = int((1 - test_split_ratio) * len(df_shuffled))

        # Create the training and testing DataFrames
        train_df = df_shuffled[:split_index]
        test_df = df_shuffled[split_index:]

        # --- 4. SAVE THE FINAL CSV FILES ---
        train_df.to_csv(train_output_path, index=False)
        print(f"\nFinal training data saved to: {train_output_path} ({len(train_df)} rows)")

        test_df.to_csv(test_output_path, index=False)
        print(f"Final testing data saved to: {test_output_path} ({len(test_df)} rows)")

    except FileNotFoundError:
        print(f"Error: The file '{os.path.basename(input_path)}' was not found.")
        print("Please run your main data processing script first to generate it.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# This block allows you to run the script directly from the terminal
if __name__ == "__main__":
    split_final_data()