import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_ram_vs_price():
    """
    Loads the raw laptop training data, cleans the 'Ram' column,
    and generates a scatter plot to visualize the relationship between
    RAM and Price.
    """
    try:
        # --- 1. DEFINE FILE PATH AND LOAD DATA ---
        input_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'train_data.csv')
        print(f"Loading data from: {input_path}")
        df = pd.read_csv(input_path)

        # --- 2. CLEAN THE NECESSARY COLUMNS ---
        # We only need to clean 'Ram' for this specific plot.
        # Remove 'GB' and convert the column to an integer type.
        if 'Ram' in df.columns and df['Ram'].dtype == 'object':
            df['Ram'] = df['Ram'].str.replace('GB', '').astype(int)
        
        # Ensure 'Price' is a numeric type
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        df.dropna(subset=['Ram', 'Price'], inplace=True) # Drop rows if conversion failed

        # --- 3. GENERATE THE SCATTER PLOT ---
        print("Generating scatter plot...")
        
        # Use seaborn for a more visually appealing plot
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.figure(figsize=(10, 6))

        sns.scatterplot(x='Ram', y='Price', data=df, alpha=0.6)

        # --- 4. CUSTOMIZE AND SHOW THE PLOT ---
        plt.title('Relationship between RAM and Laptop Price', fontsize=16)
        plt.xlabel('RAM (in GB)', fontsize=12)
        plt.ylabel('Price', fontsize=12)
        
        # Set x-axis ticks to show actual RAM values
        plt.xticks(sorted(df['Ram'].unique()))

        print("Displaying plot. Close the plot window to end the script.")
        plt.show()

    except FileNotFoundError:
        print(f"Error: The file 'train_data.csv' was not found in the 'data' directory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    plot_ram_vs_price()