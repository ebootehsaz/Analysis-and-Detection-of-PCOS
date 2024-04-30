import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from constants import PCOS_processed_filepath
from utils import load_data

def explore_data(filepath):
    # Load the dataset
    df = load_data(filepath)
    
    # Calculate correlations
    correlation_matrix = df.corr()
    
    # Get the correlations of features with the target variable 'PCOS (Y/N)'
    target_correlations = correlation_matrix['PCOS (Y/N)'].sort_values(ascending=False)
    
    # Filter out correlations that might be too weak to be of practical significance
    significant_correlation_threshold = 0.3  # Threshold can be adjusted based on domain knowledge or significance level
    significant_correlations = target_correlations[(target_correlations.abs() > significant_correlation_threshold) & (target_correlations.index != 'PCOS (Y/N)')]
    
    # Print the features that have significant correlation with the target
    print("Recommended features to use based on correlation with PCOS (Y/N):")
    print(significant_correlations)

if __name__ == '__main__':
    explore_data(PCOS_processed_filepath)
