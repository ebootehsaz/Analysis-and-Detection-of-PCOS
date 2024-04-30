import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from constants import PCOS_processed_filepath
from utils import load_data


def plot_heatmap(correlation_matrix, title, figsize=(10, 8)):
    """ Plot a heatmap given a correlation matrix. """
    plt.figure(figsize=figsize)
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title(title)
    plt.show()


def get_correlations(filepath, threshold=0.2):
    # Load the dataset
    df = load_data(filepath)
    
    # Calculate correlations with PCOS (Y/N)
    correlations = df.corr()['PCOS (Y/N)'].sort_values(ascending=False)

    significant_correlations = correlations[(correlations.abs() > threshold) & (correlations.index != 'PCOS (Y/N)')]
    
    # Print recommended features based on correlation
    print("Recommended features to use based on correlation with PCOS (Y/N):")
    print(significant_correlations)    
    
    return df, significant_correlations


def explore_data(df, significant_correlations):
    # Filter the correlation matrix to show only significant correlations
    significant_features = significant_correlations.index.tolist()
    significant_features.append('PCOS (Y/N)')  # Add the target variable to the list for visualization
    filtered_corr_matrix = df.corr().loc[significant_features, significant_features]

    # Plotting the heatmap of filtered correlation matrix
    plot_heatmap(filtered_corr_matrix, 'Heatmap of Significantly Correlated Features with PCOS (Y/N)')


def explore_data_single_correlation(significant_correlations):
    significant_correlations_df = pd.DataFrame(significant_correlations)
    plot_heatmap(significant_correlations_df, 'Significant Correlation with PCOS (Y/N)', figsize=(4, 8))


def plot_follicle_vs_pcos(df):
    # Calculate the mean Follicle No. (R) for each PCOS category
    mean_follicles = df.groupby('PCOS (Y/N)')['Follicle No. (R)'].mean()

    # Plotting the results as a bar chart
    plt.figure(figsize=(8, 6))
    sns.barplot(x=mean_follicles.index, y=mean_follicles.values)
    plt.title('Average Follicle No. (R) by PCOS Status')
    plt.xlabel('PCOS Diagnosis')
    plt.ylabel('Average Follicle No. (R)')
    plt.xticks(ticks=[0, 1], labels=['No PCOS', 'PCOS'])  # Set custom labels for clarity
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    df, significant_correlations = get_correlations(PCOS_processed_filepath)
    # explore_data(df, significant_correlations)
    # explore_data_single_correlation(significant_correlations)
    plot_follicle_vs_pcos(df)