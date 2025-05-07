#%% MODULE BEGINS
module_name = 'Water Quality Feature Extraction'

'''
Version: v2.3

Description:
    Feature extraction module for water quality classification project.
    Extracts statistical features from preprocessed data.

Authors:
    Abdullah Al Mofael, Fahim Muntasir Rabbi

Date Created     : 2024-04-05
Date Last Updated: 2025-05-02

Doc:
    Input: ./CODE/INPUT/TRAIN/preprocessed_data.xlsx
    Output: ./CODE/OUTPUT/extracted_features.xlsx
    Plots: ./CODE/OUTPUT/feature_plots/
'''

#%% IMPORTS                    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict

#%% PATH CONSTANTS             ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CODE_DIR = Path(__file__).resolve().parent
INPUT_FILE = CODE_DIR / "INPUT" / "TRAIN" / "preprocessed_data.xlsx"
OUTPUT_FILE = CODE_DIR / "OUTPUT" / "extracted_features.xlsx"
PLOTS_DIR = CODE_DIR / "OUTPUT" / "feature plots"

WATER_PARAMETERS = [
    'PH', 'EC', 'DO', 'BOD', 'COD', 'TDS', 'SS', 'TS',
    'Chlorine', 'Total Alkalinity', 'Turbidity',
    'NH3-N', 'NO2-N', 'NO3-N', 'Phosphate'
]

#%% CONFIGURATION              ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def initialize_directories() -> None:
    """Create required directory structure."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def configure_visualization() -> None:
    """Configure plotting settings."""
    plt.style.use('seaborn-v0_8')
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'figure.figsize': (12, 8),
        'font.size': 12
    })

#%% FEATURE EXTRACTION         ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def calculate_features(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate mean, absolute, and log features for each parameter."""
    features_list = []
    for _, sample in data.iterrows():
        sample_features = {}
        for param in WATER_PARAMETERS:
            if param in data.columns:
                value = sample[param]
                if not pd.isna(value):
                    sample_features.update({
                        f"{param}_mean": value,
                        f"{param}_abs": abs(value),
                        f"{param}_log": np.log(value) if value > 0 else 0
                    })
        features_list.append(sample_features)
    features_df = pd.DataFrame(features_list)
    features_df.index.name = "Sample"
    return features_df

def save_features(features_dict: Dict[str, pd.DataFrame], output_path: Path) -> bool:
    """Save feature dictionary to an Excel file."""
    try:
        with pd.ExcelWriter(output_path) as writer:
            for sheet_name, features in features_dict.items():
                features.to_excel(writer, sheet_name=sheet_name, index=False)
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save features: {e}")
        return False

#%% VISUALIZATION              ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def plot_feature_distributions(features: pd.DataFrame) -> None:
    """Plot and save boxplot of feature distributions."""
    try:
        plt.figure()
        features.plot(kind='box')
        plt.title('Feature Distributions')
        plt.xticks(rotation=45)
        plt.tight_layout()
        output_path = PLOTS_DIR / "feature_distributions.png"
        plt.savefig(output_path, dpi=300)
        plt.close()
    except Exception as e:
        print(f"[ERROR] Boxplot plotting failed: {e}")

def plot_feature_correlations(features: pd.DataFrame) -> None:
    """Plot and save feature correlation heatmap."""
    try:
        corr = features.corr()
        plt.figure(figsize=(14, 10))
        sns.heatmap(corr, cmap='coolwarm', center=0, annot=False)
        plt.title("Feature Correlation Heatmap")
        plt.tight_layout()
        output_path = PLOTS_DIR / "feature_correlations.png"
        plt.savefig(output_path, dpi=300)
        plt.close()
    except Exception as e:
        print(f"[ERROR] Correlation plot failed: {e}")

#%% MAIN PROCESS               ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def main() -> bool:
    """Run the full feature extraction pipeline."""
    initialize_directories()
    configure_visualization()

    if not INPUT_FILE.exists():
        print(f"[ERROR] Input file not found at: {INPUT_FILE}")
        return False

    try:
        data_sheets = pd.read_excel(INPUT_FILE, sheet_name=None)
        features_dict = {}

        for sheet_name, data in data_sheets.items():
            features = calculate_features(data)
            features_dict[sheet_name] = features
            if sheet_name == 'Training':
                plot_feature_distributions(features)
                plot_feature_correlations(features)

        if not save_features(features_dict, OUTPUT_FILE):
            return False

        print("[STATUS] Feature extraction completed successfully.")
        print(f"[STATUS] Features saved to: {OUTPUT_FILE}")
        print(f"[STATUS] Plots saved to: {PLOTS_DIR}")
        return True

    except Exception as e:
        print(f"[ERROR] Main process failed: {e}")
        return False

#%% SELF-RUN                  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
    print(f"\"{module_name}\" module begins.")
    if main():
        print("[INFO] Execution completed.")
    else:
        print("[INFO] Execution terminated with errors.")
