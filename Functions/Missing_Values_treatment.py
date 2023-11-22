# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 16:19:56 2023

@author: ParnikaPancholi
"""
import pandas as pd


def missing_value_treatment(src="", indsn="", imputation_methods=" "):
    """
    Perform missing value treatment on a indsn.
    Args:
        :param indsn: Input input_dfFrame with missing values.
         :param imputation_methods: Dictionary of variables as keys and imputation methods as values.
                                   Available methods: 'mean', 'median', 'mode', 'ffill', 'bfill', 'custom'
                                   :type src: object
        :param src:
    Returns:
        pd.input_dfFrame: input_dfFrame with missing values treated.
    """
    input_df = pd.read_pickle(r"{}\{}.pkl".format(str(src), str(indsn)))
    print(r"{} successfully read.".format(str(indsn)))
    missing_columns = input_df.columns[input_df.isnull().any()].tolist()

    # Drop columns with 99% or more missing values
    missing_threshold = 1
    columns_to_drop = [col for col in missing_columns if input_df[col].isnull().mean() >= missing_threshold]
    input_df.drop(columns_to_drop, axis=1, inplace=True)
    print(f"Dropping '{columns_to_drop}' as missing value is =100%")
    for col in missing_columns:
        print(col)
        if col in imputation_methods:
            method = imputation_methods[col]
            if method == 'Mean':
                input_df[col].fillna(input_df[col].mean(), inplace=True)
            elif method == 'Median':
                input_df[col].fillna(input_df[col].median(), inplace=True)
            elif method == 'Mode':
                input_df[col].fillna(input_df[col].mode()[0], inplace=True)
            elif method == 'Zero':
                input_df[col].fillna(0, inplace=True)
            elif method == 'outlier':
                print('outlier')
                #Calculate the 5th percentile value
                percentile_value_lower = input_df[col].quantile(0.05)

                # Calculate the 95th percentile value
                percentile_value_upper = input_df[col].quantile(0.95)
                # Cap values lower than the 5th percentile
                input_df[col] = input_df[col].clip(lower=percentile_value_lower)
                # Cap values greater than the 95th percentile
                input_df[col] = input_df[col].clip(upper=percentile_value_upper)

            else:
                print(f"Invalid imputation method specified for column '{col}'. Skipping imputation.")
        else:
            print(f"No imputation method specified for column '{col}'. Skipping imputation.")
    try:
        input_df.to_pickle(r"{}\{}_v1.pkl".format(str(src), str(indsn)))

        print("Following output files successfully saved in Outcopy folder")
        print("{}_v1".format(str(indsn)))

        return True
    except:
        return False
