# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 16:16:49 2023

@author: ParnikaPancholi
"""

from statsmodels.formula.api import ols
from statsmodels.stats.tests import WaldTest

src=r'C:\Users\ParnikaPancholi\OneDrive - Progyny\Documents\OB_not_auth\Data\Input'
# Load the data
input_df = pd.read_pickle(r"{}\{}.pkl".format(str(src), str(Ob_Noic_23_v1)))

# Specify the linear model
model = ols('y ~ x', data=input_df).fit()

# Create a Wald test object
wald_test = WaldTest(model, restrictions={'x': [1, 0]})

# Perform the Wald test
p_value = wald_test.pvalue

# Interpret the results
if p_value < 0.05:
    print('Reject the null hypothesis of linearity.')
else:
    print('Fail to reject the null hypothesis of linearity.')
    
    
    
    import statsmodels.api as sm

X_with_constant = sm.add_constant(X)  # Add a constant to the X matrix
model = sm.Logit(y, X_with_constant)
result = model.fit()

# Wald test for linearity
wald_test = result.wald_test("X")
print(wald_test.summary())



# defining the dependent and independent variables 
Xtrain = df[col] 
ytrain = data_out['events']
#building the model and fitting the data 
log_reg = sm.Logit(ytrain, Xtrain).fit() 
