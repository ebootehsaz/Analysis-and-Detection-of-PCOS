import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pandas import DataFrame

from constants import PCOS_inf_filepath, PCOS_woinf_filepath

from utils import load_data, merge_data
from process_kaggle_data import process_data

# This program contains utility functions to visualize the data


def visualize_and_remove_outliers(df: DataFrame, column: str):
    sns.boxplot(x=df[column])
    plt.title(f'Boxplot of {column} before outlier removal')
    plt.show()

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df_filtered = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    sns.boxplot(x=df_filtered[column])
    plt.title(f'Boxplot of {column} after outlier removal')
    plt.show()
    return df_filtered


def display_data(df: DataFrame):
    """
    Generate tables/graphs to display the data.
    """
    # Display the data
    print(f"Data in {df.attrs['file_path']}:", df.head())

    # Display the data types
    print(f"Data types in {df.attrs['file_path']}:", df.dtypes)

    # Display the summary statistics
    print(f"Summary statistics in {df.attrs['file_path']}:", df.describe())

    # Display the missing values
    print(f"Missing values in {df.attrs['file_path']}:", df.isnull().sum())

    # Display the unique values in each column
    for col in df.columns:
        print(f"Unique values in {col}:", df[col].unique())

    # Display the distribution of the data
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            sns.histplot(df[col])
            plt.title(f"Distribution of {col}")
            plt.show()

    # Display the correlation between the columns
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation between columns")
    plt.show()

    # Display the relationship between columns
    sns.pairplot(df)
    plt.title("Relationship between columns")
    plt.show()


def display_data2(df: DataFrame):
    """
    Generate tables/graphs to display the data.
    """
    # Display the data
    print(f"Data in {df.attrs['file_path']}:", df.head())

    # Display the data types
    print(f"Data types in {df.attrs['file_path']}:", df.dtypes)

    # Display the summary statistics
    print(f"Summary statistics in {df.attrs['file_path']}:", df.describe())

    # Display the missing values
    print(f"Missing values in {df.attrs['file_path']}:", df.isnull().sum())

    # Display the unique values in each column
    for col in df.columns:
        print(f"Unique values in {col}:", df[col].unique())

    # Display the distribution of the data
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            sns.histplot(df[col])
            plt.title(f"Distribution of {col}")
            plt.show()

    # Display the correlation between the columns
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation between columns")
    plt.show()

    # Display the relationship between columns
    sns.pairplot(df)
    plt.title("Relationship between columns")
    plt.show()


def convert_to_numeric(df, column_names):
    for column in column_names:
        df[column] = pd.to_numeric(df[column], errors='coerce')
    return df


def main():
    # Load the data
    PCOS_inf_df = load_data(PCOS_inf_filepath)
    original_inf_size = PCOS_inf_df.shape[0]
    PCOS_woinf_df = load_data(PCOS_woinf_filepath)
    original_woinf_size = PCOS_woinf_df.shape[0]

    # Process the data
    PCOS_inf_df = process_data(PCOS_inf_df)
    PCOS_woinf_df = process_data(PCOS_woinf_df)

    # Individual dataset processing, visualization, and removal of outliers
    datasets = {
        "PCOS_inf": (PCOS_inf_df, original_inf_size),
        "PCOS_woinf": (PCOS_woinf_df, original_woinf_size)
    }

    for name, (df, original_size) in datasets.items():
        numeric_columns = ['I beta-HCG(mIU/mL)', 'II beta-HCG(mIU/mL)', 'AMH(ng/mL)']
        df = convert_to_numeric(df, numeric_columns)
        for column in numeric_columns:
            df = visualize_and_remove_outliers(df, column)

        new_size = df.shape[0]
        rows_removed = original_size - new_size
        percentage_removed = (rows_removed / original_size) * 100

        print(f"{name} data size after removing outliers: {new_size} rows")
        print(f"Number of rows removed due to outliers in {name}: {rows_removed}")
        print(f"Percentage of data removed due to outliers in {name}: {percentage_removed:.2f}%")

        # Generate pie chart for each dataset
        labels = 'Remaining Data', 'Removed Data'
        sizes = [100 - percentage_removed, percentage_removed]
        colors = ['lightblue', 'lightcoral']
        explode = (0.1, 0)  # explode the 1st slice (i.e., 'Remaining Data')
        plt.figure(figsize=(7, 7))
        plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                shadow=True, startangle=140)
        plt.axis('equal')
        plt.title(f'Percentage of Data Remaining vs. Removed Due to Outliers in {name}')
        plt.show()

        print(f"{name} data size after removing outliers: {new_size} rows")
        print(f"Number of rows removed due to outliers in {name}: {rows_removed}")
        print(f"Percentage of data removed due to outliers in {name}: {percentage_removed:.2f}%")

    # Merge the data
    PCOS_df_merged = merge_data(PCOS_inf_df, PCOS_woinf_df)
    original_merged_size = PCOS_df_merged.shape[0]

    # Convert specified columns to numeric in the merged dataset
    PCOS_df_merged = convert_to_numeric(PCOS_df_merged, numeric_columns)

    # Visualize and remove outliers for the merged dataset
    for column in numeric_columns:
        PCOS_df_merged = visualize_and_remove_outliers(PCOS_df_merged, column)

    new_merged_size = PCOS_df_merged.shape[0]
    rows_removed_merged = original_merged_size - new_merged_size
    percentage_removed_merged = (rows_removed_merged / original_merged_size) * 100

    # Generate pie chart for the merged dataset
    sizes_merged = [100 - percentage_removed_merged, percentage_removed_merged]
    plt.figure(figsize=(7, 7))
    plt.pie(sizes_merged, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=140)
    plt.axis('equal')
    plt.title('Percentage of Data Remaining vs. Removed Due to Outliers in Merged Dataset')
    plt.show()

    print(f"Merged dataset data size after removing outliers: {new_merged_size} rows")
    print(f"Number of rows removed due to outliers in merged dataset: {rows_removed_merged}")
    print(f"Percentage of data removed due to outliers in merged dataset: {percentage_removed_merged:.2f}%")

    # Display the data
    # display_data(PCOS_df_merged)
    display_data(PCOS_woinf_df)
    display_data(PCOS_inf_df)
    display_data2(PCOS_inf_df)
    display_data(PCOS_df_merged)


if __name__ == '__main__':
    main()
