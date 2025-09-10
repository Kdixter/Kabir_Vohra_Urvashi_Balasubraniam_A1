import pandas as pd
import os

def split_data(test_split_ratio=0.1, random_state=42):
    """
    Loads the transformed dataset, splits it into training and testing sets,
    and saves them as separate CSV files.

    Args:
        test_split_ratio (float): The proportion of the dataset to allocate to the test set.
        random_state (int): A seed for the random number generator to ensure
                            reproducible shuffles and splits.
    """
    try:
        # --- 1. DEFINE FILE PATHS ---
        base_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        input_path = os.path.join(base_dir, 'train_data_transformed.csv')
        train_output_path = os.path.join(base_dir, 'training_set.csv')
        test_output_path = os.path.join(base_dir, 'testing_set.csv')

        # --- 2. LOAD THE TRANSFORMED DATA ---
        print(f"Loading transformed data from: {input_path}")
        df = pd.read_csv(input_path)

        # --- 3. SHUFFLE AND SPLIT THE DATA ---
        # Shuffle the DataFrame to ensure the split is random and unbiased.
        # Using a fixed random_state makes the shuffle deterministic (the same every time).
        df_shuffled = df.sample(frac=1, random_state=random_state)

        # Calculate the index at which to split the data
        split_index = int((1 - test_split_ratio) * len(df_shuffled))

        # Create the training and testing sets
        train_df = df_shuffled[:split_index]
        test_df = df_shuffled[split_index:]

        print(f"\nData successfully split:")
        print(f" - {len(train_df)} samples for the training set ({ (1-test_split_ratio)*100 }%)")
        print(f" - {len(test_df)} samples for the testing set ({ test_split_ratio*100 }%)")

        # --- 4. SAVE THE NEW CSV FILES ---
        train_df.to_csv(train_output_path, index=False)
        print(f"\nTraining set saved to: {train_output_path}")

        test_df.to_csv(test_output_path, index=False)
        print(f"Testing set saved to: {test_output_path}")

    except FileNotFoundError:
        print(f"Error: The file '{os.path.basename(input_path)}' was not found.")
        print("Please run the data_preprocessing.py script first to generate it.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Running:
if __name__ == "__main__":
    split_data()
