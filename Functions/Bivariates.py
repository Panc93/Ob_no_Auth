# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 15:30:48 2023

@author: ParnikaPancholi
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 16:40:55 2023

@author: ParnikaPancholi
"""

def bivar_exec(src="", wrk="", oup="", tnp="", indsn="", resp="", timelwlmt="",
               timeuplmt=""):
    """
    :param src: Path to input folder
    :param wrk: Path to Work folder
    :param oup: Path to Output folder
    :param cop: Path to Outcopy folder
    :param tnp: Path to Temporary folder
    :param indsn: Name of the Input Data frame
    :param resp: Name of the Target variable
    :param maxmisslmt: Missing percentage limit. Keep same value from the previous code.(1-100 Scale , Default:75)
    :param lvllmt: Max. percentage distribution of a single level of a variable permissible. (1-100 Scale , Default:99.9)
    :param timelwlmt: Specify Min Date (based on Training Data)
    :param timeuplmt: Specify Max Date (based on Training Data
    :return: Dataframes containing WOE and IV values of variables
    """
    # Check if all the input parameters are properly specified
    if src == "" or wrk == "" or oup == ""  or wrk == "" or indsn == "" or resp == "" or timelwlmt == "" or timeuplmt == "":
        print("INPUT PARAMETERS ARE NOT SPECIFIED PROPERLY IN FUNCTION CALL!")
        return False

    # import sys library for searching the below location for USD package(s)
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.preprocessing import LabelEncoder
    from scipy import stats
    from scipy.stats import fisher_exact
    import pandas as pd
    import sys
    #import statsmodels.api as sm 
  
    sys.path.insert(1, r'..\.')
   #import sys
    #sys.path.insert(1, r'..\.')
   # from Libraries.check_path import check_path
    #print("CHECK: IF SPECIFIED PATHS ARE VALID")
    #check1 = check_path(src, wrk, oup, tnp)
    #if check1:
    
    try:

        # import bin_dtls
        #bin_dtls = pd.read_pickle(r"{}\{}_bin_dtls.pkl".format(str(wrk), str(indsn)))
        # read input dataframe
        clm_dtls = pd.read_excel(r"{}\{}_clmn_dtls.xlsx".format(str(src), str(indsn[:10])))
        # read input dataframe
       
        input_df = pd.read_pickle(r"{}\{}.pkl".format(str(src), str(indsn)))

        # read binned dataframe
        #binned_df = pd.read_pickle(r"{}\{}_bin.pkl".format(str(wrk), str(indsn)))

        print("Required Dataframes successfully read.")
        print("---------------------------------------------------")
        print("")
    except:
        print("Unable to read required Dataframes!")
        return False

    if len(input_df[resp].unique()) != 2:
        print("TARGET VARIABLE DOES NOT BELONG TO BINARY CLASS!")
        return False

    # Bivariate Analysis code starts
    input_df['target_date'] = pd.to_datetime(input_df['target_date'])
    # Separate out holdout and train data
    bin_df_train = input_df[(input_df['target_date'] >= "'" + timelwlmt + "''") & (input_df['target_date'] <= "'" + timeuplmt + "''")]
    
    # preparing list of variables for which bivariate is required
    req_list = clm_dtls[clm_dtls.BIVAR_TYPE.isin(["CHAR","CHAR - OTHER", "CHAR - BINARY"])]['VAR_NAME']
   # req_list = bin_df_train.columns.drop(['target_date', 'patient_id', resp])
    len(list(req_list))

    # Chisquare test for vars Segm_Ext
    bin_bi_chisq = pd.DataFrame(columns=['VAR_NAME', 'CHI-SQUARE_VAL', 'P-VALUE', 'DOF'])
    for x in req_list:
        temp_df = pd.DataFrame()
        c_table = pd.crosstab(bin_df_train[x], bin_df_train[resp])
        a = stats.chi2_contingency(c_table)[0:3]
        temp_df['CHI-SQUARE_VAL'] = pd.Series(a[0])
        temp_df['P-VALUE'] = pd.Series(a[1])
        temp_df['DOF'] = pd.Series(a[2])
        temp_df['VAR_NAME'] = x
       # bin_bi_chisq = bin_bi_chisq.append(temp_df[['VAR_NAME', 'CHI-SQUARE_VAL', 'P-VALUE', 'DOF']])
        bin_bi_chisq = pd.concat([bin_bi_chisq, temp_df[['VAR_NAME', 'CHI-SQUARE_VAL', 'P-VALUE', 'DOF']]], ignore_index=True)
   
    bin_bi_fisher = pd.DataFrame(columns=['VAR_NAME', 'P-VALUE'])   
    binary_list = clm_dtls[clm_dtls.BIVAR_TYPE.isin([ "CHAR - BINARY"])]['VAR_NAME']
    for x in binary_list:
        # Perform Fisher's exact test on the contingency table
        c_table = pd.crosstab(bin_df_train[x], bin_df_train[resp])
        p_value = fisher_exact(c_table)[1] 
        # Add the results to the DataFrame
        bin_bi_fisher = pd.concat([bin_bi_fisher, pd.DataFrame({'VAR_NAME': [x], 'P-VALUE': [p_value]})], ignore_index=True)
    #Mutual information
    print(" MI LOGIC")
    # Assuming you have a binary target variable in your data
    target = bin_df_train[resp]
    # Initialize a dictionary to store mutual information scores
    mi_scores = {}
    # Initialize a dictionary to store label encoding information
    #label_encodings = {}
    one_hot_decisions={}
   # Iterate through the columns in column_details
    for index, row in clm_dtls.iterrows():
       column_name = row['VAR_NAME']
       BIVAR_type = row['BIVAR_TYPE']
       DEF_type = row['DEF_TYPE']
       print(column_name+ DEF_type+BIVAR_type )
       if (BIVAR_type == 'NUM'):
           # Numeric variable
           numeric_data = bin_df_train[column_name]
           mi_score = mutual_info_classif(numeric_data.values.reshape(-1, 1), target, discrete_features=False)
           mi_scores[column_name] = mi_score[0]

       elif (BIVAR_type == 'CHAR - OTHER' and DEF_type == 'NUM'):
           # Ordinal variable
             ordinal_data = bin_df_train[column_name]
             # Label encode ordinal data
             #le = LabelEncoder()
             #ordinal_data_encoded = le.fit_transform(ordinal_data)
             #label_encodings[column_name] = le.classes_.tolist()
             mi_score = mutual_info_classif(ordinal_data.values.reshape(-1, 1), target)
             mi_scores[column_name] = mi_score[0]
             
       elif (BIVAR_type == 'CHAR - OTHER' and DEF_type == 'CHAR'):
            # Categorical variable (one-hot encoding)
            categorical_data = bin_df_train[column_name]
            # One-hot encode and save encoding decisions
            categorical_data_encoded = pd.get_dummies(categorical_data, prefix=column_name)
            one_hot_decisions[column_name] = categorical_data_encoded.columns.tolist()
            mi_score = mutual_info_classif(categorical_data_encoded, target, discrete_features=True)
            mi_scores[column_name] = sum(mi_score)

       elif (BIVAR_type == 'CHAR - BINARY'):
           # Binary variable (assuming no encoding required)
           binary_data = bin_df_train[column_name]
           mi_score = mutual_info_classif(binary_data.values.reshape(-1, 1), target, discrete_features=True)
           mi_scores[column_name] = mi_score[0]

      #Print or use mi_scores as needed
    mi_scores_df = pd.DataFrame(list(mi_scores.items()), columns=['Variable', 'Mutual Information Score'])
    
    Clmn_list = clm_dtls[clm_dtls.BIVAR_TYPE.isin(["CHAR","CHAR - OTHER", "CHAR - BINARY","NUM"])]['VAR_NAME']
    if mi_scores_df.shape[0]== len(Clmn_list):
        print("Mutual Information Scores calculates for all {} variables".format(str(len(Clmn_list))))
    else: 
        print("columns do not match")
        return mi_scores_df
    # Create a DataFrame for the mutual information scores

    print(" MI LOGIC END")
     
    
    # Saving Output
    try:
    
    # Save the output to Excel and pickle files
           # Save the DataFrame to a file
        bin_bi_fisher.to_excel(r"{}\{}_fisher.xlsx".format(str(wrk), str(indsn)))
        mi_scores_df.to_pickle(r"{}\{}_mi_scores.pkl".format(str(wrk), str(indsn)))
        mi_scores_df.to_excel(r"{}\{}_mi_scores.xlsx".format(str(wrk), str(indsn)))
        #label_encodings.to_excel(r"{}\{}_label_encoding.xlsx".format(str(wrk), str(indsn)))
        # Save the label encoding information in a pickle file
        #bin_bi_iv.to_pickle(r"{}\{}_bi_iv.pkl".format(str(cop), str(indsn)))
        #one_hot_decisions_df.to_excel(r"{}\{}_encoding_decisions.xlsx".format(str(wrk), str(indsn)))
        
        bin_bi_chisq.to_pickle(r"{}\{}_bi_chisq.pkl".format(str(wrk), str(indsn)))
        bin_bi_chisq.to_excel(r"{}\{}_bi_chisq.xlsx".format(str(wrk), str(indsn)))
        
        print("Following output files successfully saved in Outcopy folder")
        print("{}_bi_chisq".format(str(indsn)))
        return True
    except:
        print("Failed to save the output. Please check your output path!")
        return False