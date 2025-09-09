import pandas as pd
import os

def load_life_expectancy_data():
    """
    Loads the life expectancy dataset and diagnoses file path issues.
    """
    try:
        # --- DIAGNOSTIC CODE ADDED ---
        # Get the directory where this script is located
        script_directory = os.path.dirname(__file__)
        print(f"DEBUG: This script is located in: {os.path.abspath(script_directory)}")

        # Get the current working directory from where you are running the script
        current_working_directory = os.getcwd()
        print(f"DEBUG: You are running this script from: {current_working_directory}")
        
        # Construct the relative path
        relative_path = os.path.join(script_directory, '..', 'data', 'train_data.csv')
        
        # Get the full absolute path that Python is trying to open
        absolute_file_path = os.path.abspath(relative_path)
        print(f"DEBUG: The script is trying to open this exact file path:\n--> {absolute_file_path}\n")
        # --- END OF DIAGNOSTIC CODE ---

        # Load the CSV file into a pandas DataFrame
        data = pd.read_csv(absolute_file_path)
        
        print("Data loaded successfully!")
        print("Shape of the dataset:", data.shape)
        print("\nFirst 5 rows of the dataset:")
        print(data.head())
        
        return data

    except FileNotFoundError:
        print("-------------------------------------------------------------")
        print("FILE NOT FOUND. Let's fix this.")
        print("The path above is what the script tried, but that file doesn't exist.")
        print("\nCOMMON CAUSES:")
        print("  1. TYPO: Is the file name *exactly* 'train_data.csv'?")
        print("  2. LOCATION: Is 'train_data.csv' inside a 'data' folder, which is next to your 'src' folder?")
        print("  3. RUNNING LOCATION: Make sure you run the command from the main project folder, not from inside 'src'.")
        print("-------------------------------------------------------------")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

if __name__ == "__main__":
    life_expectancy_df = load_life_expectancy_data()