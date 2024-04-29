import re
import pandas as pd
import os
from pandas import DataFrame

# This program defines utility functions to load, merge, and save data.

from constants import PCOS_kaggle_filepath, PCOS_kaggle_filepath_page

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
        # remove ../ from the start of filepath
        filepath = re.sub(r'\.\./', '', filepath)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File path {filepath} does not exist.")

    
    # They data is either in csv or excel format
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
        df.attrs['file_path'] = filepath  # Storing file path as an attribute
    elif filepath.endswith('.xlsx'):
        sheet_name = PCOS_kaggle_filepath_page
        df = pd.read_excel(filepath, sheet_name)
        df.attrs['file_path'] = filepath  # Storing file path as an attribute
    else:
        raise ValueError(f"File path {filepath} is not a csv or excel file.")
    
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
        try:
            # remove ../ from the start of filepath 
            fp = re.sub(r'\.\./', '', fp)
            if os.path.exists(fp):
                overwrite = input(f"File path {fp} already exists. Do you want to overwrite it? (y/n): ")
                if overwrite.lower() != 'y':
                    print("Data not saved.")
                    return
            df.to_csv(fp, index=False)
            print(f"Data saved to {fp}")
        except Exception as e:
            # Check if the file path exists and prompt user to overwrite
            print(f"Error saving the data to {fp}. Error: {str(e)}")