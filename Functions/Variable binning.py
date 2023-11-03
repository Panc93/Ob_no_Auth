# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 16:42:58 2023

@author: ParnikaPancholi
"""

def var_bin(src="", wrk="", oup="", cop="", tnp="", indsn="", resp="", timelwlmt="", timeuplmt=""):
    """
    :param src: Path to input folder
    :param wrk: Path to Work folder
    :param oup: Path to Output folder
    :param cop: Path to Outcopy folder
    :param tnp: Path to Temporary folder
    :param indsn: Name of the Input Data frame
    :param resp: Name of the Target variable
    :param timelwlmt: Specify Min Date (based on Training Data)
    :param timeuplmt: Specify Max Date (based on Training Data)
    :return: Dataframes containing Binned data and their details
    """
    # Check if all the input parameters are properly specified
    if src == "" or wrk == "" or oup == "" or tnp == "" or indsn == "" or resp == "" or timelwlmt == "" or timeuplmt == "":
        print("INPUT PARAMETERS ARE NOT SPECIFIED PROPERLY IN FUNCTION CALL!")
        return False

    # import sys library for searching the below location for USD package(s)
    import numpy as np
    import pandas as pd
    from copy import deepcopy
    import sys
    sys.path.insert(1, r'..\.')
    
    print("CHECK: IF SPECIFIED PATHS ARE VALID")
    check1 = check_path(src, wrk, oup, cop, tnp)
    if check1:
        try:
            # import uni_per.pkl file
            uni_per_df = pd.read_pickle(r"{}\{}_uni_perc.pkl".format(str(cop), str(indsn)))
            # read pickle file
            input_df = pd.read_pickle(r"{}\{}.pkl".format(str(src), str(indsn)))
            print(r"{} and {}_uni_per successfully read.".format(str(indsn), str(indsn)))
            print("---------------------------------------------------")
            print("")
        except:
            print(r"Unable to read {}.pkl or {}_uni_per.pkl Dataframe!".format(str(indsn), str(indsn)))
            return False
    else:
        return False

    if len(input_df[resp].unique()) != 2:
        print("TARGET VARIABLE DOES NOT BELONG TO BINARY CLASS!")
        return False

    # Variable Binning Codes starts here
    # Seprate out holdout and train data

    input_df_train = input_df[(input_df['OBSERVATION_DATE'] >= "'" + timelwlmt + "''") & (
    input_df['OBSERVATION_DATE'] <= "'" + timeuplmt + "''")]
    input_df_hold = input_df[input_df['OBSERVATION_DATE'] > "'" + timeuplmt + "''"]

    # Copy ID vars
    # Prepare binned base dataset-Train
    train_bin = pd.DataFrame()
    train_bin['OBSERVATION_DATE'] = input_df_train['OBSERVATION_DATE']
    train_bin['KNID'] = input_df_train['KNID']
    train_bin[resp] = input_df_train[resp]

    # Prepare binned base dataset-Hold
    hold_bin = pd.DataFrame()
    hold_bin['OBSERVATION_DATE'] = input_df_hold['OBSERVATION_DATE']
    hold_bin['KNID'] = input_df_hold['KNID']
    hold_bin[resp] = input_df_hold[resp]

    # Creating Binned variables
    bin_dtls = pd.DataFrame(columns=['BIN', 'Lower_Limit(exc)', 'Upper_Limit(inc)', 'VAR_NAME', 'BASE_VARNAME'])
    bin_dtls_hold = pd.DataFrame(columns=['BASE_VARNAME', 'VAR_NAME', 'CUT'])
    for x in uni_per_df['VAR_NAME']:
        #Creating BM Bin
        if (int(uni_per_df[uni_per_df['VAR_NAME'] == x]['BM_FLAG'])) == 1:
            train_bin['BM_' + x] = input_df_train[x].notnull().astype(int)
            hold_bin['BM_' + x] = input_df_hold[x].notnull().astype(int)
        #Creating B10 Bin
        if (int(uni_per_df[uni_per_df['VAR_NAME'] == x]['B10_FLAG'])) == 1:
            a = pd.qcut(input_df_train[x].astype(float), 10, labels=False, retbins=True, duplicates='drop')
            train_bin['B10_' + x] = a[0] + 1
            temp_df = pd.DataFrame()
            temp_df['Lower_Limit(exc)'] = pd.Series(a[1][0:-1])
            temp_df['Upper_Limit(inc)'] = pd.Series(a[1][1:])
            temp_df['VAR_NAME'] = 'B10_' + x
            temp_df['BASE_VARNAME'] = x
            temp_df.reset_index(inplace=True)
            temp_df['index'] += 1
            temp_df.rename(columns={'index': 'BIN'}, inplace=True)
            bin_dtls = bin_dtls.append(temp_df)
            # for holdout
            d = deepcopy(a)
            d[1][-1] = np.inf
            d[1][0] = -np.inf
            temp1_df = pd.DataFrame()
            temp1_df['BASE_VARNAME'] = x
            temp1_df['VAR_NAME'] = 'B10_' + x
            temp1_df['CUT'] = [d[1]]
            bin_dtls_hold = bin_dtls_hold.append(temp1_df)
            hold_bin['B10_' + x] = pd.cut(input_df_hold[x].astype(float), d[1], labels=range(len(d[1]) - 1)).astype(
                'float')
            hold_bin['B10_' + x] += 1
        #Creating B5 Bin
        if (int(uni_per_df[uni_per_df['VAR_NAME'] == x]['B5_FLAG'])) == 1:
            a = pd.qcut(input_df_train[x].astype(float), 5, labels=False, retbins=True, duplicates='drop')
            train_bin['B5_' + x] = a[0] + 1
            temp_df = pd.DataFrame()
            temp_df['Lower_Limit(exc)'] = pd.Series(a[1][0:-1])
            temp_df['Upper_Limit(inc)'] = pd.Series(a[1][1:])
            temp_df['VAR_NAME'] = 'B5_' + x
            temp_df['BASE_VARNAME'] = x
            temp_df.reset_index(inplace=True)
            temp_df['index'] += 1
            temp_df.rename(columns={'index': 'BIN'}, inplace=True)
            bin_dtls = bin_dtls.append(temp_df)
            # for holdout
            d = deepcopy(a)
            d[1][-1] = np.inf
            d[1][0] = -np.inf
            temp1_df = pd.DataFrame()
            temp1_df['BASE_VARNAME'] = x
            temp1_df['VAR_NAME'] = 'B10_' + x
            temp1_df['CUT'] = [d[1]]
            bin_dtls_hold = bin_dtls_hold.append(temp1_df)
            hold_bin['B5_' + x] = pd.cut(input_df_hold[x].astype(float), d[1], labels=range(len(d[1]) - 1)).astype(
                'float')
            hold_bin['B5_' + x] += 1
        #Creating B4 Bin
        if (int(uni_per_df[uni_per_df['VAR_NAME'] == x]['B4_FLAG'])) == 1:
            a = pd.qcut(input_df_train[x].astype(float), 4, labels=False, retbins=True, duplicates='drop')
            train_bin['B4_' + x] = a[0] + 1
            temp_df = pd.DataFrame()
            temp_df['Lower_Limit(exc)'] = pd.Series(a[1][0:-1])
            temp_df['Upper_Limit(inc)'] = pd.Series(a[1][1:])
            temp_df['VAR_NAME'] = 'B4_' + x
            temp_df['BASE_VARNAME'] = x
            temp_df.reset_index(inplace=True)
            temp_df['index'] += 1
            temp_df.rename(columns={'index': 'BIN'}, inplace=True)
            bin_dtls = bin_dtls.append(temp_df)
            # for holdout
            d = deepcopy(a)
            d[1][-1] = np.inf
            d[1][0] = -np.inf
            temp1_df = pd.DataFrame()
            temp1_df['BASE_VARNAME'] = x
            temp1_df['VAR_NAME'] = 'B10_' + x
            temp1_df['CUT'] = [d[1]]
            bin_dtls_hold = bin_dtls_hold.append(temp1_df)
            hold_bin['B4_' + x] = pd.cut(input_df_hold[x].astype(float), d[1], labels=range(len(d[1]) - 1)).astype(
                'float')
            hold_bin['B4_' + x] += 1
        #Creating B3 Bin
        if (int(uni_per_df[uni_per_df['VAR_NAME'] == x]['B3_FLAG'])) == 1:
            a = pd.qcut(input_df_train[x].astype(float), 3, labels=False, retbins=True, duplicates='drop')
            train_bin['B3_' + x] = a[0] + 1
            temp_df = pd.DataFrame()
            temp_df['Lower_Limit(exc)'] = pd.Series(a[1][0:-1])
            temp_df['Upper_Limit(inc)'] = pd.Series(a[1][1:])
            temp_df['VAR_NAME'] = 'B3_' + x
            temp_df['BASE_VARNAME'] = x
            temp_df.reset_index(inplace=True)
            temp_df['index'] += 1
            temp_df.rename(columns={'index': 'BIN'}, inplace=True)
            bin_dtls = bin_dtls.append(temp_df)
            # for holdout
            d = deepcopy(a)
            d[1][-1] = np.inf
            d[1][0] = -np.inf
            temp1_df = pd.DataFrame()
            temp1_df['BASE_VARNAME'] = x
            temp1_df['VAR_NAME'] = 'B10_' + x
            temp1_df['CUT'] = [d[1]]
            bin_dtls_hold = bin_dtls_hold.append(temp1_df)
            hold_bin['B3_' + x] = pd.cut(input_df_hold[x].astype(float), d[1], labels=range(len(d[1]) - 1)).astype(
                'float')
            hold_bin['B3_' + x] += 1

    # reseting indexes
    bin_dtls_hold.reset_index(drop=True, inplace=True)
    bin_dtls.reset_index(drop=True, inplace=True)
    # replacing null values with -100 for TRAIN
    for x in train_bin.columns.drop(['KNID', 'OBSERVATION_DATE', resp]):
        if train_bin[x].dtype in ('int', 'int32', 'int64', 'float', 'float32', 'float64'):
            train_bin[x].fillna(-100, inplace=True)
        if train_bin[x].dtype == 'object' or train_bin[x].dtype == 'O':
            train_bin[x].fillna("-100", inplace=True)

    # replacing null values with -100 for HOLD
    for x in hold_bin.columns.drop(['KNID', 'OBSERVATION_DATE', resp]):
        if hold_bin[x].dtype in ('int', 'int32', 'int64', 'float', 'float32', 'float64'):
            hold_bin[x].fillna(-100, inplace=True)
        if hold_bin[x].dtype == 'object' or hold_bin[x].dtype == 'O':
            hold_bin[x].fillna("-100", inplace=True)

    # Appending Train and Holdout dataframes
    input_bin = train_bin.append(hold_bin)
    input_bin.sort_index(inplace=True)

    print("COUNT OF ALL NUMERIC VARIABLES IN THE BASE TABLE.")
    print("NUMERIC VARS")
    print("COUNTALL")
    print(len(uni_per_df.VAR_NAME))
    print("")
    print("----------------------------------------------------")

    print("NUMERIC VARS WITH VARIOUS MISSING VARIABLES")
    print("COUNTM + COUNT3 = NUMERIC VARS.")
    print("COUNTM")
    print(len(uni_per_df[uni_per_df['BM_FLAG'] == 1].VAR_NAME))
    print("")
    print("----------------------------------------------------")

    print("COUNT3")
    print(len(uni_per_df[uni_per_df['B3_FLAG'] == 1].VAR_NAME))
    print("")
    print("----------------------------------------------------")

    print("COUNT4")
    print(len(uni_per_df[uni_per_df['B4_FLAG'] == 1].VAR_NAME))
    print("")
    print("----------------------------------------------------")

    print("COUNT5")
    print(len(uni_per_df[uni_per_df['B5_FLAG'] == 1].VAR_NAME))
    print("")
    print("----------------------------------------------------")

    print("COUNT10")
    print(len(uni_per_df[uni_per_df['B10_FLAG'] == 1].VAR_NAME))
    print("")
    print("----------------------------------------------------")

    print("COUNTM + COUNT3 + COUNT4 + COUNT5 + COUNT10 + 3 = BUC_NUM_VARS.")
    print(input_bin.shape[1])

    # Saving Output
    try:
        input_bin.to_pickle(r"{}\{}_bin.pkl".format(str(wrk), str(indsn)))
        bin_dtls_hold.to_pickle(r"{}\{}_bin_dtls_hold.pkl".format(str(tnp), str(indsn)))
        bin_dtls.to_pickle(r"{}\{}_bin_dtls.pkl".format(str(wrk), str(indsn)))
        bin_dtls.to_excel(r"{}\{}_bin_dtls.xlsx".format(str(wrk), str(indsn)))
        print("Following output files successfully saved in Work and Temp folder")
        print("{}_bin".format(str(indsn)))
        print("{}_bin_dtls_hold".format(str(indsn)))
        print("{}_bin_dtls".format(str(indsn)))

    except:
        print("Failed to save the output. Please check your output path!")
        return False