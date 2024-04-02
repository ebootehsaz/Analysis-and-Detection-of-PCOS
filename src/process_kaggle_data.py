import re
from pandas import DataFrame

from constants import PCOS_inf_filepath, PCOS_woinf_filepath, \
PCOS_inf_processed_filepath, PCOS_woinf_processed_filepath, PCOS_merged_processed_filepath

from utils import load_data, merge_data, save_data_csv

# This program defines utility functions to process and clean data.

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
    # Print out the first 5 missing rows for each column with missing values
    # Find rows with missing data across any column
    rows_with_missing_data = df[df.isnull().any(axis=1)]

    # Display the rows with missing data if any
    if not rows_with_missing_data.empty:
        print("Rows with missing data:")
        print(rows_with_missing_data)
    else:
        print("No missing data in any row.")
    
    df = df.fillna('None')

    # Drop duplicates
    df = df.drop_duplicates()

    # Take a random sample of the data
    print(f"Sample of the data in {df.attrs['file_path']}:", df.sample(5))

    return df


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