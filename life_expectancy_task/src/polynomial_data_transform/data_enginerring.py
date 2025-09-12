import pandas as pd
import os

def create_interaction_features():
    """
    Loads the training dataset and engineers new interaction features.
    Saves the result to 'polynomial_training_set_engineered.csv'.

    The new features created are:
    1. 'Schooling_Income_Interaction': Product of 'Schooling' and
       'Income composition of resources'.
    2. 'BMI_Status_Diphtheria_Interaction': Product of ' BMI ', 'Status',
       and 'Diphtheria '.
    """
    try:
        # --- 1. DEFINE FILE PATHS (Corrected) ---
        # Using relative paths is the best practice. This will work on any machine.
        base_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        input_path = os.path.join(base_dir, 'training_set.csv')
        output_path = os.path.join(base_dir, 'polynomial_training_set_engineered.csv')

        # --- 2. LOAD THE TRAINING DATA ---
        print(f"Loading training data from: {input_path}")
        df = pd.read_csv(input_path)

        # --- 3. CREATE NEW INTERACTION FEATURES ---
        print("Engineering new interaction features...")

        # Interaction 1: Schooling * Income composition of resources
        df['Schooling_Income_Interaction'] = df['Schooling'] * df['Income composition of resources']
        print(" - Created 'Schooling_Income_Interaction' feature.")

        # Interaction 2: BMI * Status * Diphtheria
        df['BMI_Status_Diphtheria_Interaction'] = df[' BMI '] * df['Status'] * df['Diphtheria ']
        print(" - Created 'BMI_Status_Diphtheria_Interaction' feature.")

        # --- 4. SAVE THE NEWLY ENGINEERED DATASET ---
        df.to_csv(output_path, index=False)
        print(f"\nFeature engineering complete.")
        print(f"Engineered training set saved to: {output_path}")

        # Display the head of the new dataframe to show the new columns
        print("\nPreview of the new dataset with engineered features:")
        print(df.head())

    except FileNotFoundError:
        print(f"Error: The file '{os.path.basename(input_path)}' was not found.")
        print("Please ensure you have run the data splitting script first.")
    except KeyError as e:
        print(f"\nAn error occurred: A column name was not found: {e}")
        print("Please double-check the column names in your CSV file for typos or extra spaces.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# It's best practice to put the function call inside this block.
if __name__ == "__main__":
    create_interaction_features()
