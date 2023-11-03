# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 14:27:04 2023

@author: ParnikaPancholi
"""


def data_dict(src="", wrk="", oup="", tnp="", indsn="", limit=20):
    """
    :param src: Path to input folder
    :param wrk: Path to Work folder
    :param oup: Path to Output
    :param tnp: Path to Temporary folder
    :param indsn:Name of the Input Data frame
    :param limit: Distinct values limit. If a variable has < limit it will be further analysed as categorical else as continuous
    :return: Data dictionary of all variables in the output folder in both excel and pickle file
    """

    # Check if all the input parameters are properly specified
    if src == "" or wrk == "" or oup == "" or tnp == "" or indsn == "":
        print("INPUT PARAMETERS ARE NOT SPECIFIED PROPERLY IN FUNCTION CALL!")
        return False

    # import sys library for searching the below location for USD package
    import pandas as pd
    import sys
    sys.path.insert(1, r'..\.')
    # from Libraries.check_path import check_path
    print("CHECK: IF SPECIFIED PATHS ARE VALID")
    # check1 = check_path(src, wrk, oup, tnp)
    # if check1:
    # try:
    # import col_details xlsx file
    col_details = pd.read_excel(r"{}\{}_clmn_dtls.xlsx".format(str(src), str(indsn[:10])))
    # read pickle file
    input_df = pd.read_pickle(r"{}\{}.pkl".format(str(src), str(indsn)))
    print(r"{} and col_details successfully read.".format(str(indsn)))
    print("---------------------------------------------------")
    print("")
    # except:
    #   print(r"Unable to read {} dataframe or col_details.xlsx!".format(str(indsn)))
    #  return False

    # subset column details only for columns present in abt details
    pre_col_details = col_details[col_details['VAR_NAME'].isin(input_df.columns)]
    pre_col_details.reset_index(drop=True, inplace=True)

    pre_col_details = pre_col_details[
        ["VAR_NAME", "DEF_TYPE", "UNIVAR_TYPE", "BIVAR_TYPE", "TREND_APPLICABLE"]]
    new_vars = input_df.shape[1] - pre_col_details.shape[0]
    print(new_vars)
    print("FOLLOWING ", str(new_vars), " VARIABLES WERE NOT PART OF PRE_CLMN_DTLS DATAFRAME")
    print("CHECK THE ASSIGNED VALUES OF ALL UNKNOWN VARIABLES AND MAKE NECESSARY EDITS.")
    if new_vars > 0:
        # Preparing list of newly added columns from datasets
        add_col_details = pd.DataFrame()
        add_col_details['VAR_NAME'] = list(set(input_df.columns).difference(set(pre_col_details.VAR_NAME)))
        for x in list(add_col_details['VAR_NAME']):
            print(x)
        # sorting list of unidentified variables
        add_col_details.sort_values(by=['VAR_NAME'], inplace=True)
        add_col_details.reset_index(drop=True, inplace=True)
        # creating base_varname from var_name
        # add_col_details['BASE_VARNAME'] = add_col_details['VAR_NAME'].replace(regex=True, to_replace=r'([0-9]$)|(^(TUP|TDO|TMAX|TMIN|TAVG|TCFV|TSTD|TVOL|TR)[1-6]_?)',value=r'')
        # Finding out dtype of newly added columns
        df_sub = input_df[list(add_col_details['VAR_NAME'])]
        temp = df_sub.dtypes.astype(str).to_frame().reset_index()
        temp.columns = ['VAR_NAME', 'DTP']
        temp = df_sub.dtypes.astype(str).to_frame().reset_index()
        temp.columns = ['VAR_NAME', 'DTP']
        # Assigning Def_type to them based on dtype
        temp['DEF_TYPE'] = 'UNASSIGNED-VAR'
        temp['DEF_TYPE'][temp.DTP.str.contains('date', regex=False, case=False)] = 'DATE'
        temp['DEF_TYPE'][temp.DTP.str.contains('int', regex=False, case=False)] = 'NUM'
        temp['DEF_TYPE'][temp.DTP.str.contains('float', regex=False, case=False)] = 'NUM'
        temp['DEF_TYPE'][temp.DTP.str.contains('bool', regex=False, case=False)] = 'NUM'
        temp['DEF_TYPE'][temp.DTP.str.contains('object', regex=False, case=False)] = 'CHAR'
        if temp[temp.DEF_TYPE == 'UNASSIGNED-VAR'].size != 0:
            print("FOLLOWING VARIABLES ARE LEFT UNASSIGNED-UNIVAR_TYPE!")
            print(list(temp[temp['DEF_TYPE'] == 'UNASSIGNED-VAR'].VAR_NAME))
        # Merging def_type information  with additional features dataset
        add_col_details = pd.merge(add_col_details, temp[['VAR_NAME', 'DEF_TYPE']], on=['VAR_NAME'], how="inner")
        # Finding out Unique levels in a featur
        temp = df_sub.nunique().astype(int).to_frame().reset_index()
        temp.columns = ['VAR_NAME', 'UVC']
        add_col_details = pd.merge(add_col_details, temp[['VAR_NAME', 'UVC']], on=['VAR_NAME'], how="inner")
        # Assigning Univar type based on Def_type and UVC count
        add_col_details['UNIVAR_TYPE'] = 'UNASSIGNED-VAR'
        add_col_details.loc[add_col_details.DEF_TYPE == 'DATE', 'UNIVAR_TYPE'] = 'DATE'
        add_col_details.loc[(add_col_details.DEF_TYPE == 'CHAR') & (
            ~add_col_details.VAR_NAME.isin(['patient_id', 'mem_id_no_client'])), 'UNIVAR_TYPE'] = 'CHAR - OTHER'
        add_col_details.loc[(add_col_details.DEF_TYPE == 'CHAR') & (
            add_col_details.VAR_NAME.isin(['patient_id', 'mem_id_no_client'])), 'UNIVAR_TYPE'] = 'ID VAR'
        add_col_details.loc[
            (add_col_details.DEF_TYPE == 'CHAR') & (add_col_details.UVC == 2), 'UNIVAR_TYPE'] = 'CHAR - BINARY'

        add_col_details.loc[(add_col_details.DEF_TYPE == 'NUM'), 'UNIVAR_TYPE'] = 'NUM'
        add_col_details.loc[
            (add_col_details.DEF_TYPE == 'NUM') & (add_col_details.UVC == 2), 'UNIVAR_TYPE'] = 'CHAR - BINARY'
        add_col_details.loc[(add_col_details.DEF_TYPE == 'NUM') & (add_col_details.UVC != 2) & (
                add_col_details.UVC <= limit), 'UNIVAR_TYPE'] = 'CHAR - OTHER'
        # Assigning Bi-Var type based on Uni-Var type and asssigning NA. to target and status variables
        add_col_details['BIVAR_TYPE'] = add_col_details['UNIVAR_TYPE']
        add_col_details.loc[
            (add_col_details.VAR_NAME.str.contains('events', regex=False, case=False)), 'BIVAR_TYPE'] = 'NA.'
        add_col_details.loc[
            (add_col_details.VAR_NAME.str.contains('date', regex=False, case=False)), 'BIVAR_TYPE'] = 'NA.'
        # print list of features where automatic uni,bi-vartype is not assigned
        if add_col_details[add_col_details.UNIVAR_TYPE == 'UNASSIGNED-VAR'].size != 0:
            print("FOLLOWING VARIABLES ARE LEFT UNASSIGNED-BIVAR_TYPE!")
            print(list(add_col_details[add_col_details['UNIVAR_TYPE'] == 'UNASSIGNED-VAR'].VAR_NAME))
        # Logic for assigning Trend_Applicable feature in Data Dictionary based on lag months
        # temp = add_col_details.groupby('BASE_VARNAME')['VAR_NAME'].nunique().to_frame().reset_index()
        # temp.columns = ['BASE_VARNAME', 'MAX_MONTH']
        # temp['MAX_MONTH'] = temp['MAX_MONTH'] - 1.0
        # add_col_details = pd.merge(add_col_details, temp[['BASE_VARNAME', 'MAX_MONTH']], on=['BASE_VARNAME'],
        #                           how="inner")
        # add_col_details['TREND_APPLICABLE'] = 'NO'
        # add_col_details.loc[
        #   (add_col_details.UNIVAR_TYPE == 'NUM') & (add_col_details.MAX_MONTH >= 3), 'TREND_APPLICABLE'] = 'YES'

        # add_col_details['TREND_3M'] = 'NO'
        # add_col_details.loc[
        #   (add_col_details.TREND_APPLICABLE == 'YES') & (add_col_details.MAX_MONTH >= 3), 'TREND_3M'] = 'YES'

        # add_col_details['TREND_6M'] = 'NO'
        # add_col_details.loc[
        #    (add_col_details.TREND_APPLICABLE == 'YES') & (add_col_details.MAX_MONTH >= 6), 'TREND_6M'] = 'YES'
        # add_col_details.drop(['UVC'], axis=1, inplace=True)

        input_df_col_dtls = pd.concat([pre_col_details, add_col_details], ignore_index=True)
        input_df_col_dtls.sort_values(by=['VAR_NAME'], inplace=True)
        input_df_col_dtls.reset_index(drop=True, inplace=True)
    else:
        # If no new variable is added directly create Data Dictionary from ABT pre_col_details
        input_df_col_dtls = pre_col_details

    print("----------------------------------------------------------------")
    print("")
    print("OVERVIEW OF VARIABLE DETAILS. DEFAULT VARIABLE TYPES")
    temp = input_df_col_dtls.groupby(['DEF_TYPE'])['VAR_NAME'].count().to_frame().reset_index()
    temp.columns = ['DEF_TYPE', 'COUNT(VAR_NAME)']
    print(temp)
    print("")
    print("----------------------------------------------------------------")
    print("")
    print("VARIABLE CLASSIFICATION FOR UNIVARIATE ANALYSES.")
    temp = input_df_col_dtls.groupby(['UNIVAR_TYPE'])['VAR_NAME'].count().to_frame().reset_index()
    temp.columns = ['UNIVAR_TYPE', 'COUNT(VAR_NAME)']
    print(temp)
    print("")
    print("----------------------------------------------------------------")
    print("")
    print("VARIABLE CLASSIFICATION FOR BIVARIATE ANALYSES.")
    temp = input_df_col_dtls.groupby(['BIVAR_TYPE'])['VAR_NAME'].count().to_frame().reset_index()
    temp.columns = ['BIVAR_TYPE', 'COUNT(VAR_NAME)']
    print(temp)
    print("")
    print("----------------------------------------------------------------")
    print("")

    # Saving column_details as excel and pickle file in outcopy
    try:
        input_df_col_dtls.to_pickle(r"{}\{}_clmn_dtls.pkl".format(str(src), str(indsn)))
        input_df_col_dtls.to_excel(r"{}\{}_clmn_dtls.xlsx".format(str(src), str(indsn)))
        print("Following output files successfully saved in Outcopy folder")
        print("{}_clmn_dtls".format(str(indsn)))
    except:
        print("Failed to save the output. Please check your output path!")
        return False
