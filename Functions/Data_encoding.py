# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 14:51:10 2023

@author: ParnikaPancholi
"""

def Data_encoding(src="", oup="", wrk="", tnp="", indsn="", resp="",timelwlmt="", timeuplmt=""):
    
    """
    :param src: Path to input folder
    :param wrk: Path to Work folder
    :param oup: Path to Output folder
    :param tnp: Path to Temporary folder
    :param timelwlmt : training start
    :timeuplmt :training end
    :param indsn: Name of the Input Data frame
    
    :return: Dataframe with encoded variables in pickle file
    """
    from joblib import dump
    import pandas as pd
    from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
    import sys
    sys.path.insert(1, r'..\.')

    
     # Check if all the input parameters are properly specified
    if src == "" or wrk == "" or oup == ""  or tnp == "" or indsn == "" or resp == "" or timelwlmt == "" or timeuplmt == "":
         print("INPUT PARAMETERS ARE NOT SPECIFIED PROPERLY IN FUNCTION CALL!")
         return False
    #Read files 
    try:
        # read input dataframe
        clm_dtls = pd.read_excel(r"{}\{}_clmn_dtls.xlsx".format(str(src), str(indsn[:10])))
        # read input dataframe
        
        input_df = pd.read_pickle(r"{}\{}.pkl".format(str(src), str(indsn)))

        print("Required Dataframes successfully read.")
        print("---------------------------------------------------")
        print("")
    except:
        print("Unable to read required Dataframes!")
        return False

    #Drop eligibility status
    # For categorical and ordinal
    input_df['target_date'] = pd.to_datetime(input_df['target_date'])
    
    # Separate out holdout and train data
    input_df_train = input_df.loc[(input_df['target_date'] >= "'" + timelwlmt + "''") & (input_df['target_date'] <= "'" + timeuplmt + "''")]
    
    # preparing list of variables for which bivariate is required
    #All_char_list = clm_dtls[clm_dtls.BIVAR_TYPE.isin(["CHAR","CHAR - OTHER"])& clm_dtls.DEF_TYPE.isin(["CHAR","NUM"] )]['VAR_NAME']
    Char_list = clm_dtls[clm_dtls.BIVAR_TYPE.isin(["CHAR - OTHER"]) & clm_dtls.DEF_TYPE.isin(["CHAR"] )]['VAR_NAME']
    #len(All_char_list)
    len(Char_list)
    # Label encode ordinal variables
    oe = OrdinalEncoder()
    print("run")
    # apply le on categorical feature columns
    oe_encoded = oe.fit_transform( input_df_train[Char_list] )
    # Access the category mapping
    category_mapping = []
    for i, column_name in enumerate(Char_list):
        mapping = {
            'Variable Name': column_name,
            'Categories': oe.categories_[i].tolist(),
            'Labels': list(set(oe_encoded[:, i]))
        }
        category_mapping.append(mapping)

    # Convert the list of dictionaries to a DataFrame
    category_mapping_df = pd.DataFrame(category_mapping)
    encoded_df = pd.DataFrame(oe_encoded, columns=Char_list)
    # One-hot encode nominal variables
    ohe = OneHotEncoder(sparse_output=False)
    one_hot_encoded_df = pd.DataFrame(ohe.fit_transform(encoded_df),index=input_df_train.index,columns=ohe.get_feature_names_out())

    #Extract only the columns that didnt need to be encoded
    data_other_cols = input_df_train.drop(columns=Char_list)
    
    #Concatenate the two dataframes : 
    data_out = pd.concat([data_other_cols,one_hot_encoded_df], axis=1)

  
    # Save the label encoder and one-hot encoder to pickle files for holdout encodings in future
   

    try:   
        
    
        # Save the one-hot encoded dataframe
        data_out.to_pickle(r"{}\{}_v2.pkl".format(str(src), str(indsn[:10])))
       
        print("Input data has {} variables".format(str(input_df.shape)))
        print("Input train data has {} variables".format(str(input_df_train.shape)))
        print("Output data has {} variables".format(str(data_out.shape)))
        #Catergory mapping for understanding labeling mapping
        category_mapping_df.to_excel(r"{}\{}_v2_label_map.xlsx".format(str(src),str(indsn[:10])))
        # Save the label encoder and one-hot encoder to pickle files for holdout encodings in future
        dump(oe, r"{}\{}_v2_label_encoder.joblib".format((str(src)), str(indsn[:10])))
        dump(ohe, r"{}\{}_v2_one_hot_encoder.joblib".format((str(src)), str(indsn[:10])))
        print("Files saved")
        return True
    except:
        print("Files not saved. Check paths")
        return False
    # Load the label encoder and one-hot encoder
    #with open('output/label_encoder.pkl', 'rb') as f:
     # label_encoder = pickle.load(f)
    
    #with open('output/one_hot_encoder.pkl', 'rb') as f:
     # one_hot_encoder = pickle.load(f)
    
    # Encode the categorical variables in the holdout data
    
    #holdout_df[categorical_columns] = label_encoder.transform(holdout_df[categorical_columns])
    #holdout_encoded_df = one_hot_encoder.transform(holdout_df[categorical_columns])

# Use the encoded holdout data for training or testing your machine learning model
    

    