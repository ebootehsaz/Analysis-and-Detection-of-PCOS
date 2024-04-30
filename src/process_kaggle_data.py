import re
from pandas import DataFrame

from constants import PCOS_processed_filepath, PCOS_kaggle_filepath

from utils import load_data, save_data_csv

# This program defines utility functions to process and clean data.


def process_data(df: DataFrame) -> DataFrame:
    # Checking column names, missing values, duplicates
    print(f"Columns in {df.attrs['file_path']}:", df.columns)
    print(f"Missing values in {df.attrs['file_path']}:\n{df.isnull().sum()}")
    print(f"Duplicates in {df.attrs['file_path']}: {df.duplicated().sum()}")

    # Dropping repeated/unnecessary columns
    df = df.drop(['Unnamed: 44', 'Sl. No', 'Patient File No.'], axis=1, errors='ignore')

    # Renaming column due to misspelling in original df
    df.rename(columns={'Marraige Status (Yrs)': 'Marriage Status (Yrs)'}, inplace=True, errors='ignore')

    # Fix column names - optional
    df.columns = df.columns.str.strip()  # .str.replace(' ', '_').str.lower()
    df.columns = [re.sub(r'\s+', ' ', col).strip() for col in df.columns]

    # Remove leading/trailing whitespaces from the data, remove trailing commas, periods
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df = df.applymap(lambda x: x.rstrip('.') if isinstance(x, str) else x)

    # Fix missing values
    # Print out the first 5 missing rows for each column with missing values
    # Find rows with missing data across any column
    rows_with_missing_data = df[df.isnull().any(axis=1)]
    lst_missing_columns = df.columns[df.isna().any()].tolist()

    print(f"There are a total of {len(rows_with_missing_data)} rows with missing data in {df.attrs['file_path']}.")

    # filling missing values with their median
    for x in lst_missing_columns:
        df[x] = df[x].fillna(df[x].median())  # filling columns with missing value with their median

    # Verifying if any missing values are left
    if not len((df.columns[df.isna().any()].tolist())):
        print(f"Columns with missing data: {df.columns[df.isna().any()].tolist()}")

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
    PCOS_df = load_data(PCOS_kaggle_filepath)

    # Process the data
    PCOS_df = process_data(PCOS_df)

    # Save the processed data
    save_data_csv(PCOS_df, PCOS_processed_filepath)


if __name__ == '__main__':
    main()
