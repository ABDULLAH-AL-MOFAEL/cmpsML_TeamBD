#%% MODULE BEGINS
module_name = 'Water Quality Preprocessing'

'''
Version: v1.0

Description:
    Preprocessing module for water quality dataset. Handles missing values,
    outliers, feature scaling, and data splitting for machine learning.

Authors:
    Abdullah Al Mofael, Fahim Muntasir Rabbi

Date Created     : 2024-03-28
Date Last Updated: 2024-03-29

Doc:
    This module preprocesses water quality data for ML classification.
    Input: WQ_dataset_ML_project.xlsx
    Output: preprocessed_data.xlsx with Training, Validation, Testing sheets

Notes:
    - Uses median imputation for missing values
    - IQR method for outlier handling
    - Z-score standardization
    - 60-20-20 train-val-test split
'''

#%% IMPORTS                    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
    import os
    #os.chdir("./../..")
#

#custom imports
from   copy       import deepcopy as dpcpy
from pathlib import Path

#other imports
import matplotlib.pyplot as plt
import numpy             as np 
import pandas            as pd
import seaborn           as sns
from   sklearn.model_selection import train_test_split
from   sklearn.preprocessing  import StandardScaler


#%% CONSTANTS                   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DATA_PATH  = BASE_DIR / "CODE" / "INPUT" / "Raw Data" / "RawData.xlsx"  
OUTPUT_DATA_PATH = BASE_DIR / "CODE" / "INPUT" / "TRAIN" / "preprocessed_data.xlsx"
PLOTS_DIR        = BASE_DIR / "CODE" / "OUTPUT" / "feature plots"
TRAIN_RATIO      = 0.6
VAL_RATIO        = 0.2
TEST_RATIO       = 0.2
RANDOM_STATE     = 42


#%% CONFIGURATION               ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Create output directory if not exists
if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)


#%% INITIALIZATIONS             ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# (Initializations done within functions)


#%% DECLARATIONS                ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Function definitions Start Here

def load_data(file_path):
    """Load water quality data from Excel file.
    
    Args:
        file_path (str): Path to input Excel file
        
    Returns:
        pd.DataFrame: Raw water quality data
    """
    return pd.read_excel(file_path, sheet_name='Sheet1')
#

def handle_missing_values(dataframe):
    """Handle missing values using median imputation.
    
    Args:
        dataframe (pd.DataFrame): Input data with missing values
        
    Returns:
        pd.DataFrame: Data with missing values imputed
    """
    return dataframe.fillna(dataframe.median())
#

def remove_outliers(dataframe):
    """Detect and handle outliers using IQR method.
    
    Args:
        dataframe (pd.DataFrame): Input data with potential outliers
        
    Returns:
        pd.DataFrame: Data with outliers replaced by median
    """
    df_clean = dpcpy(dataframe)
    
    numeric_cols = df_clean.select_dtypes(include=np.number).columns
    
    for col in numeric_cols:
        q1 = df_clean[col].quantile(0.25)
        q3 = df_clean[col].quantile(0.75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        median_val = df_clean[col].median()
        df_clean[col] = np.where((df_clean[col] < lower_bound) | 
                              (df_clean[col] > upper_bound), 
                              median_val, df_clean[col])
    return df_clean
#

def standardize_features(dataframe):
    """Standardize features using Z-score normalization.
    
    Args:
        dataframe (pd.DataFrame): Input data to be standardized
        
    Returns:
        pd.DataFrame: Standardized data
        StandardScaler: Fitted scaler object
    """
    numeric_cols = dataframe.select_dtypes(include=np.number).columns
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(dataframe[numeric_cols])
    
    df_scaled = dpcpy(dataframe)
    df_scaled[numeric_cols] = scaled_values
    return df_scaled, scaler
#

def split_data(dataframe):
    """Split data into training, validation and test sets.
    
    Args:
        dataframe (pd.DataFrame): Input data to be split
        
    Returns:
        tuple: (train_df, val_df, test_df)
    """
    # First split to separate out test set
    train_val, test = train_test_split(
        dataframe, 
        test_size=TEST_RATIO, 
        random_state=RANDOM_STATE
    )
    
    # Second split to separate train and validation
    train, val = train_test_split(
        train_val,
        test_size=VAL_RATIO/(TRAIN_RATIO + VAL_RATIO),
        random_state=RANDOM_STATE
    )
    
    return train, val, test
#

def save_data(train_df, val_df, test_df, output_path):
    """Save processed data to Excel file.
    
    Args:
        train_df (pd.DataFrame): Training data
        val_df (pd.DataFrame): Validation data
        test_df (pd.DataFrame): Test data
        output_path (str): Path to output Excel file
    """
    with pd.ExcelWriter(output_path) as writer:
        train_df.to_excel(writer, sheet_name='Training', index=False)
        val_df.to_excel(writer, sheet_name='Validation', index=False)
        test_df.to_excel(writer, sheet_name='Testing', index=False)
#

def generate_visualizations(dataframe, output_dir):
    """Generate and save preprocessing visualizations.
    
    Args:
        dataframe (pd.DataFrame): Input data for visualization
        output_dir (str): Directory to save plots
    """
    # Missing values heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(dataframe.isnull(), cbar=False, cmap='viridis')
    plt.title("Missing Values Heatmap")
    plt.savefig(os.path.join(output_dir, "missing_values_heatmap.png"))
    plt.close()
    
    # Boxplots of key features
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=dataframe[['BOD', 'Turbidity', 'DO']])
    plt.title("Feature Distributions with Outliers")
    plt.savefig(os.path.join(output_dir, "boxplots_pre_outlier.png"))
    plt.close()
    
    # Correlation matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(dataframe.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
    plt.close()
#

def main():
    """Main preprocessing pipeline."""
    # Load raw data
    raw_data = load_data(INPUT_DATA_PATH)
    
    # Handle missing values
    data_no_missing = handle_missing_values(raw_data)
    
    # Remove outliers
    data_no_outliers = remove_outliers(data_no_missing)
    
    # Standardize features
    data_scaled, _ = standardize_features(data_no_outliers)
    
    # Split data
    train, val, test = split_data(data_scaled)
    
    # Save processed data
    save_data(train, val, test, OUTPUT_DATA_PATH)
    
    # Generate visualizations
    generate_visualizations(raw_data, PLOTS_DIR)
    
    print("Preprocessing completed successfully.")
#


#%% MAIN CODE                  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Main code start here


#%% SELF-RUN                   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Main Self-run block
if __name__ == "__main__":
    
    print(f"\"{module_name}\" module begins.")
    
    #TEST Code
    main()