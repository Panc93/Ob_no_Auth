# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 18:12:14 2023

@author: ParnikaPancholi
"""


def univar_exec(src="", wrk="", oup="", tnp="", indsn="", Train_start_dt="", Train_end_dt="", minmisslmt="",
                l1misslmt="", l2misslmt="", maxmisslmt=75):
    """
    :param wrk:
    :param l2misslmt:
    :param maxmisslmt:
    :param minmisslmt:
    :param l1misslmt:
    :param src: Path to input folder
    :param oup: Path to Output folder
    :param tnp: Path to Temporary folder
    :param indsn: Name of the Input Data frame
    :param Train_start_dt: Specify Min Date (based on Training Data)
    :param Train_end_dt: Specify Max Date (based on Training Data)
    :return: Univariate Analysis of all variables in Data Dictionary indicating missing percentages,value_counts etc.
    """
    # Check if all the input parameters are properly specified
    if (
            src == "" or wrk == "" or oup == "" or tnp == "" or indsn == "" or Train_start_dt == "" or Train_end_dt == "" or maxmisslmt == "" or minmisslmt == "" or l1misslmt == "" or l2misslmt == ""):
        print("INPUT PARAMETERS ARE NOT SPECIFIED PROPERLY IN FUNCTION CALL!")
        return False

    # import sys library for searching the below location for USD package(s)
    import pandas as pd
    import sys
    sys.path.insert(1, r'..\.')
    from datetime import date
    ###from Libraries.check_path import check_path
    # print("CHECK: IF SPECIFIED PATHS ARE VALID")
    # check1 = check_path(src, oup, tnp)
    # if check1:
    #   try:
    # import data dictionary for base variables
    col_dtls = pd.read_excel(r"{}\{}_clmn_dtls.xlsx".format(str(src), str(indsn[:10])))
    # read input pickle file
    input_df = pd.read_pickle(r"{}\{}.pkl".format(str(src), str(indsn)))
    #   print(r"{} and {}_clmn_dtls successfully read.".format(str(indsn), str(indsn)))
    #    print("---------------------------------------------------")
    #     print("")
    #  except:
    #       print(r"Unable to read {} or {}_clmn_dtls Dataframe!".format(str(indsn), str(indsn)))
    #        return False
    # else:
    #      return False

    # UNIVARIATE CODE STARTS FROM HERE
    col_dtls['VAR_NAME'] = col_dtls['VAR_NAME'].str.lower()
    # Seprate out holdout and train data
    if len(col_dtls.VAR_NAME) == input_df.shape[1]:
        print("TOTAL NUMBER OF VARIABLES")
        print(len(col_dtls.VAR_NAME))
        print("")
    else:
        print("NO OF COLUMNS IN INPUT AND COLUMN DETAILS DATAFRAME DOES NOT MATCH. CHECK DATA DICT DATAFRAME!")
        print(len(col_dtls.VAR_NAME))
        print(input_df.shape[1])
        return False
    # Convert the str objects to datetime.date objects
    Train_start_dt = date.fromisoformat(Train_start_dt)
    Train_end_dt = date.fromisoformat(Train_end_dt)
    input_df_train = input_df[(input_df['target_date'] >= Train_start_dt) & (input_df['target_date'] <= Train_end_dt)]
    input_df_hold = input_df[input_df['target_date'] > Train_end_dt]

    # Resetting index
    input_df.reset_index(drop=True, inplace=True)
    input_df_train.reset_index(drop=True, inplace=True)
    input_df_hold.reset_index(drop=True, inplace=True)

    # Univariate representing missing value counts
    input_df_uni_miss = pd.DataFrame()
    input_df_uni_miss['NMISS'] = input_df_train.isnull().sum()
    input_df_uni_miss['MISSING_PERC'] = (input_df_uni_miss['NMISS'] / len(input_df_train)) * 100
    input_df_uni_miss.reset_index(inplace=True)
    input_df_uni_miss.rename(columns={'index': 'VAR_NAME'}, inplace=True)
    input_df_uni_miss.head()

    if len(input_df_uni_miss.VAR_NAME) == input_df.shape[1]:
        print("TOTAL NUMBER OF VARIABLES WITH MISSING VALUE ANALYSIS")
        print(len(input_df_uni_miss.VAR_NAME))
        print("")
    else:
        print(
            "NO OF COLUMNS IN INPUT DATAFRAME AND MISSING COLUMN DATAFRAME DOES NOT MATCH. CHECK DATA DICT DATAFRAME!")
        return False

    print("TOTAL LIST OF VARIABLES WITH UNIVARIATE ANALYSIS")
    print(len(col_dtls[col_dtls.UNIVAR_TYPE.isin(['VARCHAR', 'CHAR - BINARY', 'CHAR - OTHER', 'NUM'])].VAR_NAME))
    print("")

    # Univariate for categorical variables
    var_cat_list = col_dtls[col_dtls.UNIVAR_TYPE.isin(['VARCHAR', 'CHAR - BINARY', 'CHAR - OTHER'])].VAR_NAME
    uni_freq1 = pd.DataFrame(columns=['LEVELS', 'FREQUENCY', 'PERCENT', 'VAR_NAME'])
    for x in var_cat_list:
        t1 = pd.DataFrame()
        t1['FREQUENCY'] = input_df_train[x].value_counts(dropna=False)
        t1['PERCENT'] = (t1['FREQUENCY'] / len(input_df_train)) * 100
        t1['VAR_NAME'] = x
        t1.reset_index(inplace=True)
        t1.rename(columns={'index': 'LEVELS'}, inplace=True)
        uni_freq1 = pd.concat([uni_freq1, t1], axis=0)

    input_df_uni_freq = pd.merge(col_dtls[['VAR_NAME']], uni_freq1, how='inner', on='VAR_NAME')

    if input_df_uni_freq.VAR_NAME.nunique() == len(var_cat_list):
        print("TOTAL LIST OF VARIABLES WITH CHARACTER UNIVARIATE ANALYSIS")
        print(len(var_cat_list))
        print("")
    else:
        print("NO OF COLUMNS IN INPUT DATAFRAME AND UNI_FREQ DATAFRAME DOES NOT MATCH. CHECK DATA DICT DATAFRAME!")
        return False

    # Univariate for numerical variables
    var_num_list = col_dtls[col_dtls.UNIVAR_TYPE.isin(['NUM'])].VAR_NAME
    input_df_uni_perc = pd.DataFrame(
        columns=['VAR_NAME', 'N', 'NMISS', 'MISSING_PERCENT', 'MEAN', 'STD', 'MEDIAN', 'SUM', 'SKEWNESS', 'KURTOSIS',
                 'MIN', 'MAX', 'P-01', 'P-02', 'P-03', 'P-04', 'P-05', 'P-10', 'P-15', 'P-20', 'P-25', 'P-30', 'P-35',
                 'P-40', 'P-45', 'P-50', 'P-55', 'P-60', 'P-65', 'P-70', 'P-75', 'P-80', 'P-85', 'P-90', 'P-95', 'P-96',
                 'P-97', 'P-98', 'P-99', 'P-100'])
    for x in var_num_list:
        t1 = pd.DataFrame()
        t1.loc[0, 'VAR_NAME'] = x
        t1.loc[0, 'N'] = input_df_train[x].count()
        t1.loc[0, 'NMISS'] = input_df_train[x].isnull().sum()
        t1.loc[0, 'MISSING_PERCENT'] = (t1.loc[0, 'NMISS'] / (t1.loc[0, 'N'] + t1.loc[0, 'NMISS'])) * 100
        t1.loc[0, 'MEAN'] = input_df_train[x].mean(skipna=True)
        t1.loc[0, 'STD'] = input_df_train[x].std(skipna=True)
        t1.loc[0, 'MEDIAN'] = input_df_train[x].median(skipna=True)
        t1.loc[0, 'SUM'] = input_df_train[x].sum(skipna=True)
        t1.loc[0, 'SKEWNESS'] = input_df_train[x].skew(skipna=True)
        t1.loc[0, 'KURTOSIS'] = input_df_train[x].kurt(skipna=True)
        t1.loc[0, 'MIN'] = input_df_train[x].min(skipna=True)
        t1.loc[0, 'MAX'] = input_df_train[x].max(skipna=True)
        t1.loc[0, 'P-01'] = input_df_train[x].quantile(0.01)
        t1.loc[0, 'P-02'] = input_df_train[x].quantile(0.02)
        t1.loc[0, 'P-03'] = input_df_train[x].quantile(0.03)
        t1.loc[0, 'P-04'] = input_df_train[x].quantile(0.04)
        t1.loc[0, 'P-05'] = input_df_train[x].quantile(0.05)
        t1.loc[0, 'P-10'] = input_df_train[x].quantile(0.10)
        t1.loc[0, 'P-15'] = input_df_train[x].quantile(0.15)
        t1.loc[0, 'P-20'] = input_df_train[x].quantile(0.20)
        t1.loc[0, 'P-25'] = input_df_train[x].quantile(0.25)
        t1.loc[0, 'P-30'] = input_df_train[x].quantile(0.30)
        t1.loc[0, 'P-35'] = input_df_train[x].quantile(0.35)
        t1.loc[0, 'P-40'] = input_df_train[x].quantile(0.40)
        t1.loc[0, 'P-45'] = input_df_train[x].quantile(0.45)
        t1.loc[0, 'P-50'] = input_df_train[x].quantile(0.50)
        t1.loc[0, 'P-55'] = input_df_train[x].quantile(0.55)
        t1.loc[0, 'P-60'] = input_df_train[x].quantile(0.60)
        t1.loc[0, 'P-65'] = input_df_train[x].quantile(0.65)
        t1.loc[0, 'P-70'] = input_df_train[x].quantile(0.70)
        t1.loc[0, 'P-75'] = input_df_train[x].quantile(0.75)
        t1.loc[0, 'P-80'] = input_df_train[x].quantile(0.80)
        t1.loc[0, 'P-85'] = input_df_train[x].quantile(0.85)
        t1.loc[0, 'P-90'] = input_df_train[x].quantile(0.90)
        t1.loc[0, 'P-95'] = input_df_train[x].quantile(0.95)
        t1.loc[0, 'P-96'] = input_df_train[x].quantile(0.96)
        t1.loc[0, 'P-97'] = input_df_train[x].quantile(0.97)
        t1.loc[0, 'P-98'] = input_df_train[x].quantile(0.98)
        t1.loc[0, 'P-99'] = input_df_train[x].quantile(0.99)
        t1.loc[0, 'P-100'] = input_df_train[x].quantile(1)
        input_df_uni_perc = pd.concat([input_df_uni_perc, t1], axis=0)

    # Binning details based on maxmimum, l1, l2 , minimum missing percentage specified as an input
    input_df_uni_perc.loc[input_df_uni_perc['MISSING_PERCENT'] > maxmisslmt, 'BM_FLAG'] = 1
    input_df_uni_perc.loc[input_df_uni_perc['MISSING_PERCENT'] <= maxmisslmt, 'BM_FLAG'] = 0

    input_df_uni_perc.loc[input_df_uni_perc['MISSING_PERCENT'] > maxmisslmt, 'B3_FLAG'] = 0
    input_df_uni_perc.loc[input_df_uni_perc['MISSING_PERCENT'] <= maxmisslmt, 'B3_FLAG'] = 1

    input_df_uni_perc.loc[input_df_uni_perc['MISSING_PERCENT'] > l1misslmt, 'B4_FLAG'] = 0
    input_df_uni_perc.loc[input_df_uni_perc['MISSING_PERCENT'] <= l1misslmt, 'B4_FLAG'] = 1

    input_df_uni_perc.loc[input_df_uni_perc['MISSING_PERCENT'] > l2misslmt, 'B5_FLAG'] = 0
    input_df_uni_perc.loc[input_df_uni_perc['MISSING_PERCENT'] <= l2misslmt, 'B5_FLAG'] = 1

    input_df_uni_perc.loc[input_df_uni_perc['MISSING_PERCENT'] > minmisslmt, 'B10_FLAG'] = 0
    input_df_uni_perc.loc[input_df_uni_perc['MISSING_PERCENT'] <= minmisslmt, 'B10_FLAG'] = 1

    input_df_uni_perc.reset_index(drop=True, inplace=True)
    input_df_uni_perc.reset_index(drop=True, inplace=True)

    if len(input_df_uni_perc.VAR_NAME) == len(var_num_list):
        print("TOTAL NUMBER OF VARIABLES WITH NUMERICAL UNIVARIATE ANALYSIS")
        print(len(input_df_uni_perc.VAR_NAME))
        print("")
    else:
        print("NO OF COLUMNS IN INPUT DATAFRAME AND UNI_PERC DATAFRAME DOES NOT MATCH. CHECK DATA DICT DATAFRAME!")
        return False

    if len(input_df_uni_perc.VAR_NAME) + input_df_uni_freq.VAR_NAME.nunique() == len(
            col_dtls[col_dtls.UNIVAR_TYPE.isin(['VARCHAR', 'CHAR - BINARY', 'CHAR - OTHER', 'NUM'])].VAR_NAME):
        # Save the outputs
        try:
            input_df_uni_miss.to_pickle(r"{}\{}_uni_miss.pkl".format(str(oup), str(indsn)))
            input_df_uni_miss.to_excel(r"{}\{}_uni_miss.xlsx".format(str(oup), str(indsn)))

            input_df_uni_freq.to_pickle(r"{}\{}_uni_freq.pkl".format(str(oup), str(indsn)))
            input_df_uni_freq.to_excel(r"{}\{}_uni_freq.xlsx".format(str(oup), str(indsn)))

            input_df_uni_perc.to_pickle(r"{}\{}_uni_perc.pkl".format(str(oup), str(indsn)))
            input_df_uni_perc.to_excel(r"{}\{}_uni_perc.xlsx".format(str(oup), str(indsn)))

            print("Following output files successfully saved in Outcopy folder")
            print("{}_uni_miss".format(str(indsn)))
            print("{}_uni_freq".format(str(indsn)))
            print("{}_uni_perc".format(str(indsn)))

        except:
            print("Failed to save the output. Please check your output path!")

    else:
        print("MISMATCH OF COLUMNS IN UNIVARIATE ANALYSIS WITH TOTAL COLUMNS USED FOR UNIVARIATE ANALYSIS FROM INPUT DATAFRAME. CHECK DATA DICT DATAFRAME!")
        return False
