import pandas as pd
import os

def scale_engineered_features():
    """
    Loads the engineered training dataset, applies Min-Max scaling to a
    range of [-1, 1] on most features, and applies a custom mapping to the
    'Status' feature. The scaled data is then saved to a new CSV file.
    """
    try:
        # --- 1. DEFINE FILE PATHS ---
        base_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        input_path = os.path.join(base_dir, 'polynomial_training_set_engineered.csv')
        output_path = os.path.join(base_dir, 'training_set_scaled.csv')

        # --- 2. LOAD THE ENGINEERED TRAINING DATA ---
        print(f"Loading engineered data from: {input_path}")
        df = pd.read_csv(input_path)

        # --- 3. SEPARATE COLUMNS FOR DIFFERENT TRANSFORMATIONS ---
        # We separate the 'Status' column to apply a custom rule to it.
        features_to_scale = df.drop(columns=['Country', 'Life expectancy ', 'Status'])
        status_column = df['Status']
        non_features = df[['Country', 'Life expectancy ']]

        # --- 4. APPLY [-1, 1] SCALING TO MOST FEATURES ---
        print("Applying Min-Max scaling to the range [-1, 1]...")
        # The formula for scaling to [-1, 1] is: 2 * (x - min) / (max - min) - 1
        df_scaled_features = 2 * (features_to_scale - features_to_scale.min()) / (features_to_scale.max() - features_to_scale.min()) - 1

        # --- 5. APPLY CUSTOM MAPPING TO THE 'STATUS' FEATURE ---
        print("Applying custom mapping to the 'Status' feature...")
        # Original values in the preprocessed data were {Developing: 0.5, Developed: 1.0}
        # New mapping: {0.5 -> -0.5, 1.0 -> 0.5}
        df_status_mapped = status_column.map({0.5: -0.5, 1.0: 0.5})
        print(" - Mapped 'Status' values of 0.5 to -0.5 and 1.0 to 0.5.")


        # --- 6. RECOMBINE AND SAVE THE DATA ---
        # Combine the unscaled columns, the custom-mapped status, and the newly scaled features
        final_df = pd.concat([non_features, df_status_mapped, df_scaled_features], axis=1)

        final_df.to_csv(output_path, index=False)
        print(f"\nScaling complete.")
        print(f"Scaled training set saved to: {output_path}")

        # Display a preview of the scaled data
        print("\nPreview of the scaled dataset:")
        print(final_df.head())
        print("\nVerifying the new range of the 'Status' column:")
        print(f"  Unique values in 'Status': {final_df['Status'].unique()}")

    except FileNotFoundError:
        print(f"Error: The file '{os.path.basename(input_path)}' was not found.")
        print("Please run the engineer_features.py script first to generate the required file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    scale_engineered_features()