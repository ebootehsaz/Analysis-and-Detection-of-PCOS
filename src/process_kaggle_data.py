import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pandas import DataFrame


# Path to the Kaggle data
PCOS_inf_filepath = "../Kaggle_Data/PCOS_infertility.csv"
PCOS_woinf_filepath, page = "../Kaggle_Data/PCOS_data_without_infertility.xlsx", "Full_new"

# Path to save the processed data
PCOS_inf_processed_filepath = "../data/PCOS_infertility_processed.csv"
PCOS_woinf_processed_filepath = "../data/PCOS_woinf_processed.csv"
PCOS_merged_processed_filepath = "../data/PCOS_merged_processed.csv"

# Load the data into a DataFrame
def load_data(filepath: str) -> DataFrame:
    """
    Load data from a file path into a DataFrame.
    Args:
        filepath: str - file path to the data
    Returns:
        df - data loaded into a DataFrame
    """
    # Check if the file path exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File path {filepath} does not exist.")
    
    # They data is either in csv or excel format
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
        df.attrs['file_path'] = filepath  # Storing file path as an attribute
    elif filepath.endswith('.xlsx'):
        df = pd.read_excel(filepath, sheet_name=page)
        df.attrs['file_path'] = filepath  # Storing file path as an attribute
    else:
        raise ValueError(f"File path {filepath} is not a csv or excel file.")
    
    return df

def process_data(df: DataFrame) -> DataFrame:
    # Checking column names, missing values, duplicates
    print(f"Columns in {df.attrs['file_path']}:", df.columns)
    print(f"Missing values in {df.attrs['file_path']}:\n{df.isnull().sum()}")
    print(f"Duplicates in {df.attrs['file_path']}: {df.duplicated().sum()}")

    #Dropping repeated/unnecessary columns
    df = df.drop(['Unnamed: 44','Sl. No_wo', 'PCOS (Y/N)_wo', '  I   beta-HCG(mIU/mL)_wo','II    beta-HCG(mIU/mL)_wo', 'AMH(ng/mL)_wo'], axis=1, errors='ignore')

    #Renaming column due to misspelling in original df
    df.rename(columns={'Marraige Status (Yrs)': 'Marriage Status (Yrs)'}, inplace=True, errors='ignore')

    # Fix column names - optional
    df.columns = df.columns.str.strip() # .str.replace(' ', '_').str.lower()
    df.columns = [re.sub(r'\s+', ' ', col).strip() for col in df.columns]


    # Fix missing values
    # TODO: Print out the unique values in each column and how many missing values are in each column
    print('data bootcamp')
    
    df = df.fillna('None')

    # Drop duplicates
    df = df.drop_duplicates()

    # Take a random sample of the data
    print(f"Sample of the data in {df.attrs['file_path']}:", df.sample(5))

    return df

def merge_data(df1: DataFrame, df2: DataFrame) -> DataFrame:
    """
    Merge two DataFrames.
    Args:
        df1: DataFrame - first DataFrame to merge
        df2: DataFrame - second DataFrame to merge
    Returns:
        DataFrame - merged DataFrame
    """
    # print(df1.columns)
    # print(df2.columns)
    df = pd.merge(df1, df2, on='Patient File No.', suffixes=('', '_wo'), how='left')

    # remove columns if all rows of that column are NaN
    df = df.dropna(axis=1, how='all')

    return df


def save_data_csv(df:DataFrame , fp: str):
    """
    Save the data to a csv file.
    Args:
        df: DataFrame - data to save to a csv file
        fp: str - file path to save the data
    """
    # Check if the file path exists and prompt user to overwrite
    if os.path.exists(fp):
        overwrite = input(f"File path {fp} already exists. Do you want to overwrite it? (y/n): ")
        if overwrite.lower() != 'y':
            print("Data not saved.")
            return

    # Save the data
    try:
        df.to_csv(fp, index=False)
        print(f"Data saved to {fp}")
    except Exception as e:
        print(f"Error saving the data to {fp}. Error: {str(e)}")

def main():
    # Load the data
    PCOS_inf_df = load_data(PCOS_inf_filepath)
    PCOS_woinf_df = load_data(PCOS_woinf_filepath)

    # Process the data
    PCOS_inf_df = process_data(PCOS_inf_df)
    PCOS_woinf_df = process_data(PCOS_woinf_df)

    # Merge the data
    PCOS_df_merged = merge_data(PCOS_inf_df, PCOS_woinf_df)

    # Save the processed data
    save_data_csv(PCOS_df_merged, PCOS_merged_processed_filepath)
    save_data_csv(PCOS_woinf_df, PCOS_woinf_processed_filepath) 
    save_data_csv(PCOS_inf_df, PCOS_inf_processed_filepath)

if __name__ == '__main__':
    main()  