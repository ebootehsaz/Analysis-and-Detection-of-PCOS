import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pandas import DataFrame

from constants import PCOS_inf_filepath, PCOS_woinf_filepath
from utils import load_data, merge_data, save_data_csv
from process_kaggle_data import process_data
from visualize_data import convert_to_numeric, display_data


def feature_heatmap(df: DataFrame):
    df = fillna_median(df)
    plt.figure(figsize=(20,20))
    feature_plot = sns.heatmap(df.corr(), cmap= "Wistia", annot=True)
    plt.title( "Dataset Feature Correlation to PCOS" )
    plt.show()
    
    # checks correlation of features against PCOS (Y/N) column
    lst_corr = df.corr()["PCOS (Y/N)"].sort_values(ascending=False)
    return lst_corr

def fillna_median(df: DataFrame) -> DataFrame:
    # utils.py merge_data method needs to recalculate NA columns instead of dropping (TO-DO)
    # process_kaggle_data method process_data fills NA rows with "None"

    #find columns with missing values
    for col in df.columns:
        col_none = df.columns[col == ["None"]].tolist()[1]
        df.loc[col_none, "None"] = df[col].median()
    return df

def main():
    # this is taken from visualize_data.py
    PCOS_inf_df = load_data(PCOS_inf_filepath)
    original_inf_size = PCOS_inf_df.shape[0]
    PCOS_woinf_df = load_data(PCOS_woinf_filepath)
    original_woinf_size = PCOS_woinf_df.shape[0]

    PCOS_inf_df = process_data(PCOS_inf_df)
    PCOS_woinf_df = process_data(PCOS_woinf_df)

    # Individual dataset processing
    datasets = {
        "PCOS_inf": (PCOS_inf_df, original_inf_size),
        "PCOS_woinf": (PCOS_woinf_df, original_woinf_size)
    }

    for name, (df, original_size) in datasets.items():
        numeric_columns = ['I beta-HCG(mIU/mL)', 'II beta-HCG(mIU/mL)', 'AMH(ng/mL)']
        df = convert_to_numeric(df, numeric_columns)

# CALL EVERYTHING

# Merge and edit the data
    PCOS_df_merged = merge_data(PCOS_inf_df, PCOS_woinf_df)
    PCOS_df_merged = convert_to_numeric(PCOS_df_merged, numeric_columns)
    PCOS_df_merged = fillna_median(PCOS_df_merged)

    feature_heatmap(PCOS_df_merged)


# GENERATE MY HEATMAP PLEASE
    dfa = PCOS_df_merged
    dfa = fillna_median(df)
    plt.figure(figsize=(20,20))
    feature_plot = sns.heatmap(df.corr(), cmap= "Wistia", annot=True)
    plt.show()
    print(dfa)

if __name__ == '__main__':
    main()