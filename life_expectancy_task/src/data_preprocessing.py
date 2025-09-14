import os
import pandas as pd
import numpy as np
from typing import Tuple
import random


def load_data(data_path: str) -> pd.DataFrame: # same as in previous task
    
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Original data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    return df


# Drop entries where Life expectancy is missing.
def clean_life_expectancy(df: pd.DataFrame) -> pd.DataFrame:

    print("Cleaning Life expectancy column...") # to know this function has been called
    initial_count = len(df)
    
    # Drop entries where Life expectancy is missing
    df = df.dropna(subset=['Life expectancy '])
    
    final_count = len(df)
    removed_count = initial_count - final_count
    print(f"Removed {removed_count} entries with missing Life expectancy")
    print(f"Remaining entries: {final_count}")
    
    return df

# One-hot-encode "Country" feature (crucial)
def encode_country_feature(df: pd.DataFrame) -> pd.DataFrame:

    print("One-hot encoding Country feature...")
    
    # Get unique countries
    unique_countries = df['Country'].unique()
    print(f"Number of unique countries: {len(unique_countries)}")
    
    # One-hot encode Country
    country_dummies = pd.get_dummies(df['Country'], prefix='Country')
    
    # Drop original Country column and add one-hot encoded columns
    df = df.drop('Country', axis=1)
    df = pd.concat([df, country_dummies], axis=1)
    
    print(f"Added {len(country_dummies.columns)} country features")
    print(f"New data shape after country encoding: {df.shape}")
    
    return df

# Encode Status feature: 0.5 for Developed, -0.5 for Developing.
def encode_status_feature(df: pd.DataFrame) -> pd.DataFrame:
   
    print("Encoding Status feature...")
    
    # Map Status values
    status_mapping = {'Developed': 0.5, 'Developing': -0.5}
    df['Status'] = df['Status'].map(status_mapping)
    
    print(f"Status mapping: {status_mapping}")
    print(f"Status value counts after encoding:")
    print(df['Status'].value_counts())
    
    return df

# Handle missing Adult Mortality values.
     #If missing, take median of that country's prior Adult Mortality values,
    # otherwise take median of entire column.
def handle_adult_mortality(df: pd.DataFrame) -> pd.DataFrame:
   
    print("Handling Adult Mortality missing values...")
    
    missing_count = df['Adult Mortality'].isna().sum()
    print(f"Missing Adult Mortality values: {missing_count}")
    
    if missing_count > 0:
        # Sort by Country and Year for proper ordering
        df_sorted = df.sort_values(['Country', 'Year']).copy()
        
        # For each missing value, try to get country-specific median first
        for idx in df_sorted[df_sorted['Adult Mortality'].isna()].index:
            country = df_sorted.loc[idx, 'Country']
            year = df_sorted.loc[idx, 'Year']
            
            # Get prior years' Adult Mortality for this country
            country_data = df_sorted[
                (df_sorted['Country'] == country) & 
                (df_sorted['Year'] < year) & 
                (df_sorted['Adult Mortality'].notna())
            ]
            
            if len(country_data) > 0:
                # Use median of country's prior values
                median_value = country_data['Adult Mortality'].median()
                df_sorted.loc[idx, 'Adult Mortality'] = median_value
                print(f"Filled missing Adult Mortality for {country} ({year}) with country median: {median_value}")
            else:
                # Use overall column median
                overall_median = df['Adult Mortality'].median()
                df_sorted.loc[idx, 'Adult Mortality'] = overall_median
                print(f"Filled missing Adult Mortality for {country} ({year}) with overall median: {overall_median}")
        
        df = df_sorted.sort_index()  # Restore original order
    
    return df

# Handle missing Alcohol values by taking median of entire column. 
def handle_alcohol(df: pd.DataFrame) -> pd.DataFrame:
    
    print("Handling Alcohol missing values...")
    
    missing_count = df['Alcohol'].isna().sum()
    print(f"Missing Alcohol values: {missing_count}")
    
    if missing_count > 0:
        median_value = df['Alcohol'].median()
        df['Alcohol'] = df['Alcohol'].fillna(median_value)
        print(f"Filled {missing_count} missing Alcohol values with median: {median_value}")
    
    return df

# Drop entire rows where BMI is missing as this is a crucial feature
def handle_bmi(df: pd.DataFrame) -> pd.DataFrame:
    
    print("Handling BMI missing values...")
    
    initial_count = len(df)
    df = df.dropna(subset=[' BMI '])
    final_count = len(df)
    removed_count = initial_count - final_count
    
    print(f"Removed {removed_count} rows with missing BMI")
    print(f"Remaining entries: {final_count}")
    
    return df

# Handle missing Polio values by taking median of entire column.
def handle_polio(df: pd.DataFrame) -> pd.DataFrame:
    
    print("Handling Polio missing values...")
    
    missing_count = df['Polio'].isna().sum()
    print(f"Missing Polio values: {missing_count}")
    
    if missing_count > 0:
        median_value = df['Polio'].median()
        df['Polio'] = df['Polio'].fillna(median_value)
        print(f"Filled {missing_count} missing Polio values with median: {median_value}")
    
    return df

#Handle missing Total expenditure values by taking mean of entire column.
def handle_total_expenditure(df: pd.DataFrame) -> pd.DataFrame:
    
    print("Handling Total expenditure missing values...")
    
    missing_count = df['Total expenditure'].isna().sum()
    print(f"Missing Total expenditure values: {missing_count}")
    
    if missing_count > 0:
        mean_value = df['Total expenditure'].mean()
        df['Total expenditure'] = df['Total expenditure'].fillna(mean_value)
        print(f"Filled {missing_count} missing Total expenditure values with mean: {mean_value}")
    
    return df

# Handle missing Diphtheria values by taking mean of entire column.
def handle_diphtheria(df: pd.DataFrame) -> pd.DataFrame:
    
    print("Handling Diphtheria missing values...")
    
    missing_count = df['Diphtheria '].isna().sum()
    print(f"Missing Diphtheria values: {missing_count}")
    
    if missing_count > 0:
        mean_value = df['Diphtheria '].mean()
        df['Diphtheria '] = df['Diphtheria '].fillna(mean_value)
        print(f"Filled {missing_count} missing Diphtheria values with mean: {mean_value}")
    
    return df

# Take previous year's GDP for that country, if not available take mean of entire column.
def handle_gdp(df: pd.DataFrame) -> pd.DataFrame:
    print("Handling GDP missing values...")
    
    missing_count = df['GDP'].isna().sum()
    print(f"Missing GDP values: {missing_count}")
    
    if missing_count > 0:
        # Sort by Country and Year for proper ordering
        df_sorted = df.sort_values(['Country', 'Year']).copy()
        
        # For each missing value, try to get previous year's GDP for that country
        for idx in df_sorted[df_sorted['GDP'].isna()].index:
            country = df_sorted.loc[idx, 'Country']
            year = df_sorted.loc[idx, 'Year']
            
            # Get previous year's GDP for this country
            prev_year_data = df_sorted[
                (df_sorted['Country'] == country) & 
                (df_sorted['Year'] == year - 1) & 
                (df_sorted['GDP'].notna())
            ]
            
            if len(prev_year_data) > 0:
                # Use previous year's GDP
                prev_gdp = prev_year_data['GDP'].iloc[0]
                df_sorted.loc[idx, 'GDP'] = prev_gdp
                print(f"Filled missing GDP for {country} ({year}) with previous year: {prev_gdp}")
            else:
                # Use overall column mean
                overall_mean = df['GDP'].mean()
                df_sorted.loc[idx, 'GDP'] = overall_mean
                print(f"Filled missing GDP for {country} ({year}) with overall mean: {overall_mean}")
        
        df = df_sorted.sort_index()  # Restore original order
    
    return df


def handle_thinness_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values for thinness features by taking median of entire column.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with thinness features missing values filled
    """
    print("Handling thinness features missing values...")
    
    thinness_features = [' thinness  1-19 years', ' thinness 5-9 years']
    
    for feature in thinness_features:
        missing_count = df[feature].isna().sum()
        print(f"Missing {feature} values: {missing_count}")
        
        if missing_count > 0:
            median_value = df[feature].median()
            df[feature] = df[feature].fillna(median_value)
            print(f"Filled {missing_count} missing {feature} values with median: {median_value}")
    
    return df


def handle_income_composition(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop entries where Income composition of resources is missing.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with Income composition missing entries removed
    """
    print("Handling Income composition of resources missing values...")
    
    initial_count = len(df)
    df = df.dropna(subset=['Income composition of resources'])
    final_count = len(df)
    removed_count = initial_count - final_count
    
    print(f"Removed {removed_count} entries with missing Income composition of resources")
    print(f"Remaining entries: {final_count}")
    
    return df


def handle_schooling(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing Schooling values by taking mean of entire column.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with Schooling missing values filled
    """
    print("Handling Schooling missing values...")
    
    missing_count = df['Schooling'].isna().sum()
    print(f"Missing Schooling values: {missing_count}")
    
    if missing_count > 0:
        mean_value = df['Schooling'].mean()
        df['Schooling'] = df['Schooling'].fillna(mean_value)
        print(f"Filled {missing_count} missing Schooling values with mean: {mean_value}")
    
    return df


def drop_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop specified features: Hepatitis B, Measles, Population.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with specified features removed
    """
    print("Dropping specified features...")
    
    features_to_drop = ['Hepatitis B', 'Measles ', 'Population']
    
    for feature in features_to_drop:
        if feature in df.columns:
            df = df.drop(feature, axis=1)
            print(f"Dropped feature: {feature}")
        else:
            print(f"Feature not found: {feature}")
    
    print(f"Data shape after dropping features: {df.shape}")
    return df


def normalize_features(df: pd.DataFrame, target_column: str = 'Life expectancy ') -> pd.DataFrame:
    """
    Normalize all features (except target) to range [-1, 1].
    
    Args:
        df: Input DataFrame
        target_column: Name of the target column
        
    Returns:
        DataFrame with normalized features
    """
    print("Normalizing features to [-1, 1] range...")
    
    # Get feature columns (exclude target)
    feature_columns = [col for col in df.columns if col != target_column]
    
    print(f"Normalizing {len(feature_columns)} features")
    
    # Normalize each feature to [-1, 1] range
    for column in feature_columns:
        if df[column].dtype in ['int64', 'float64']:  # Only normalize numeric columns
            min_val = df[column].min()
            max_val = df[column].max()
            
            if max_val != min_val:  # Avoid division by zero
                # Normalize to [0, 1] first, then scale to [-1, 1]
                df[column] = 2 * (df[column] - min_val) / (max_val - min_val) - 1
                print(f"Normalized {column}: [{min_val:.2f}, {max_val:.2f}] -> [-1, 1]")
            else:
                print(f"Skipped {column}: constant value")
    
    return df


def split_data(df: pd.DataFrame, test_size: float = 0.075, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train and test sets.
    
    Args:
        df: Input DataFrame
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, test_df)
    """
    print(f"Splitting data: {1-test_size:.1%} train, {test_size:.1%} test")
    
    # Set random seed
    random.seed(random_state)
    np.random.seed(random_state)
    
    # Shuffle the data
    df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Calculate split index
    n_samples = len(df_shuffled)
    test_samples = int(n_samples * test_size)
    
    # Split the data
    test_df = df_shuffled[:test_samples]
    train_df = df_shuffled[test_samples:]
    
    print(f"Train set: {len(train_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    
    return train_df, test_df


def save_processed_data(train_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: str) -> None:
    """
    Save processed train and test data to CSV files.
    
    Args:
        train_df: Training DataFrame
        test_df: Testing DataFrame
        output_dir: Output directory path
    """
    print("Saving processed data...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output paths
    train_path = os.path.join(output_dir, 'life_expectancy_train_processed.csv')
    test_path = os.path.join(output_dir, 'life_expectancy_test_processed.csv')
    
    # Save to CSV
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Training data saved to: {train_path}")
    print(f"Test data saved to: {test_path}")
    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")


def main():
    """Main function to perform all data preprocessing steps."""
    
    # Define paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_path = os.path.join(project_root, 'data', 'life_expectancy.csv')
    output_dir = os.path.join(project_root, 'data')
    
    print("=" * 80)
    print("LIFE EXPECTANCY DATA PREPROCESSING")
    print("=" * 80)
    
    try:
        # Load data
        df = load_data(data_path)
        
        # Step 1: Clean Life expectancy
        df = clean_life_expectancy(df)
        
        # Step 2: Encode Status feature (before dropping Country)
        df = encode_status_feature(df)
        
        # Step 3: Handle missing values for Adult Mortality (needs Country column)
        df = handle_adult_mortality(df)
        
        # Step 4: Handle missing values for GDP (needs Country column)
        df = handle_gdp(df)
        
        # Step 5: One-hot encode Country (after all country-specific operations)
        df = encode_country_feature(df)
        
        # Step 6: Handle missing values for Alcohol
        df = handle_alcohol(df)
        
        # Step 7: Drop Hepatitis B and Measles features
        df = drop_features(df)
        
        # Step 8: Handle BMI (drop rows with missing BMI)
        df = handle_bmi(df)
        
        # Step 9: Handle missing values for Polio
        df = handle_polio(df)
        
        # Step 10: Handle missing values for Total expenditure
        df = handle_total_expenditure(df)
        
        # Step 11: Handle missing values for Diphtheria
        df = handle_diphtheria(df)
        
        # Step 12: Drop Population feature (already done in drop_features)
        
        # Step 13: Handle thinness features
        df = handle_thinness_features(df)
        
        # Step 14: Handle Income composition (drop rows with missing values)
        df = handle_income_composition(df)
        
        # Step 15: Handle missing values for Schooling
        df = handle_schooling(df)
        
        # Step 16: Normalize all features to [-1, 1] range
        df = normalize_features(df)
        
        # Step 17: Split data into train/test (92.5%/7.5%)
        train_df, test_df = split_data(df, test_size=0.075, random_state=42)
        
        # Step 18: Save processed data
        save_processed_data(train_df, test_df, output_dir)
        
        print("\n" + "=" * 80)
        print("DATA PREPROCESSING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        # Print final summary
        print(f"\nFinal Summary:")
        print(f"Original data shape: {df.shape}")
        print(f"Training data shape: {train_df.shape}")
        print(f"Test data shape: {test_df.shape}")
        print(f"Features (excluding target): {train_df.shape[1] - 1}")
        print(f"Target column: Life expectancy ")
        
    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")
        raise


if __name__ == "__main__":
    main()
