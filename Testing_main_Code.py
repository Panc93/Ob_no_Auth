# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 18:56:22 2023

@author: ParnikaPancholi
"""

import pandas as pd

from Functions.Bivariates import bivar_exec
from Functions.Data_dictionary import data_dict
from Functions.Data_encoding import Data_encoding

import pickle as pk

from Functions.Missing_Values_treatment import missing_value_treatment
from Functions.Univariate_code import univar_exec

query = "SELECT * FROM  Financebi_db.panc.OB_NOIC_ADS_ASOF0923F "
Ob_Noic_23 = pd.read_sql_query(query, con=Snowflakeconnection())

file_path = 'Input/Ob_Noic_23.pkl'
with open(file_path, 'wb') as file:
    pk.dump(Ob_Noic_23, file)

data_dict(src=r'C:\Users\ParnikaPancholi\PycharmProjects\Ob_no_Auth\Data\Input',
              wrk=r'C:\Users\ParnikaPancholi\PycharmProjects\Ob_no_Auth\Data\Work',
              oup=r'C:\Users\ParnikaPancholi\PycharmProjects\Ob_no_Auth\Data\Output', tnp='Temp',
              indsn='Ob_Noic_23', limit=10)

# Numeric value imputation. For future incorporate it in data dict file
imputation_methods = {'deductible': 'Median', 'coinsurance': 'Median', 'moop': 'Median',
                      'w_btwn_coverage_onboard': 'Median', 'smart_cycles_allowed_by_plan': 'Median'}

missing_value_treatment(src=r'C:\Users\ParnikaPancholi\PycharmProjects\Ob_no_Auth\Data\Input',
                        indsn='Ob_Noic_23',
                        imputation_methods=imputation_methods)

# Creating data dict based on updated variables


# Doing the univariate analysis
univar_exec(src=r'C:\Users\ParnikaPancholi\PycharmProjects\Ob_no_Auth\Data\Input',
              wrk=r'C:\Users\ParnikaPancholi\PycharmProjects\Ob_no_Auth\Data\Work',
              oup=r'C:\Users\ParnikaPancholi\PycharmProjects\Ob_no_Auth\Data\Output',
              tnp='Temp',
            indsn='Ob_Noic_23_v1',
            Train_start_dt='2021-04-01',
            Train_end_dt='2023-04-01',
            minmisslmt=10, l1misslmt=30, l2misslmt=50,
            maxmisslmt=75)

# Doing the bivariate analysis

Data_encoding(src=r'C:\Users\ParnikaPancholi\OneDrive - Progyny\Documents\OB_not_auth\Data\Input',
              oup=r'C:\Users\ParnikaPancholi\OneDrive - Progyny\Documents\OB_not_auth\Data\Output',
              wrk=r'C:\Users\ParnikaPancholi\OneDrive - Progyny\Documents\OB_not_auth\Data\Work',
              tnp=r'C:\Users\ParnikaPancholi\OneDrive - Progyny\Documents\OB_not_auth\Data\Temp',
              indsn='Ob_Noic_23_v1', resp='events', timelwlmt='2021-04-01', timeuplmt='2023-04-01')

bivar_exec(src=r'C:\Users\ParnikaPancholi\OneDrive - Progyny\Documents\OB_not_auth\Data\Input',
           oup=r'C:\Users\ParnikaPancholi\OneDrive - Progyny\Documents\OB_not_auth\Data\Output',
           wrk=r'C:\Users\ParnikaPancholi\OneDrive - Progyny\Documents\OB_not_auth\Data\Work',
           tnp=r'C:\Users\ParnikaPancholi\OneDrive - Progyny\Documents\OB_not_auth\Data\Temp',
           indsn='Ob_Noic_23_v1', resp='events', timelwlmt='2021-04-01',
           timeuplmt='2023-04-01')

Data_encoding(src=r'C:\Users\ParnikaPancholi\OneDrive - Progyny\Documents\OB_not_auth\Data\Input',
              wrk=r'C:\Users\ParnikaPancholi\OneDrive - Progyny\Documents\OB_not_auth\Data\Work',
              oup=r'C:\Users\ParnikaPancholi\OneDrive - Progyny\Documents\OB_not_auth\Data\Temp',
              tnp=r'C:\Users\ParnikaPancholi\OneDrive - Progyny\Documents\OB_not_auth\Data\Temp',
              indsn='Ob_Noic_23_v1', resp='events', timelwlmt='2021-04-01',
              timeuplmt='2023-04-01')

data_dict(src=r'C:\Users\ParnikaPancholi\OneDrive - Progyny\Documents\OB_not_auth\Data\Input',
          wrk=r'C:\Users\ParnikaPancholi\OneDrive - Progyny\Documents\OB_not_auth\Data\Work',
          oup=r'C:\Users\ParnikaPancholi\OneDrive - Progyny\Documents\OB_not_auth\Data\Output', tnp='Temp',
          indsn='Ob_Noic_23_v2', limit=10)
