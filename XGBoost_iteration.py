import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report
import shap
import joblib
from sklearn.metrics import cohen_kappa_score
# Load the data
src = r'C:\Users\ParnikaPancholi\PycharmProjects\Ob_no_Auth\Data\Input'
indsn = 'Ob_Noic_23_v2'
hl = 'Ob_Noic_23_v2_hl'
data = pd.read_pickle(r"{}\{}.pkl".format(str(src), str(indsn)))
hold_out = pd.read_pickle(r"{}\{}.pkl".format(str(src), str(hl)))

data['progyny_rx'] = data['progyny_rx'].astype('int64')
data['type_of_fully_insured_plan'] = data['type_of_fully_insured_plan'].astype('int64')
data['rx_embedded_moop'] = data['rx_embedded_moop'].astype('int64')
data['email_refused'] = data['email_refused'].astype('int64')
data['rx_embedded_deductible'] = data['rx_embedded_deductible'].astype('int64')

hold_out['progyny_rx'] = hold_out['progyny_rx'].astype('int64')
hold_out['type_of_fully_insured_plan'] = hold_out['type_of_fully_insured_plan'].astype('int64')
hold_out['rx_embedded_moop'] = hold_out['rx_embedded_moop'].astype('int64')
hold_out['email_refused'] = data['email_refused'].astype('int64')
hold_out['rx_embedded_deductible'] = hold_out['rx_embedded_deductible'].astype('int64')
hold_out = hold_out.drop(columns=[col for col in hold_out.columns if col.startswith('current_pharmacy')])
drop_columns = ['mem_id_no_client', 'patient_id', 'onboard_date', 'target_date', 'events','days_since_last_callog']


data['tot_previous_treatments'] = data['previous_treatments_cp']+data['previous_treatments_ivf']+data['previous_treatments_tst']+data['previous_treatments_ds'] + data['previous_treatments_iui']+data['previous_treatments_fet']+data['previous_treatments_onc']+data['previous_treatments_pgt']+data['previous_treatments_ado']
hold_out['tot_previous_treatments'] = hold_out['previous_treatments_cp']+hold_out['previous_treatments_ivf']+hold_out['previous_treatments_tst']+hold_out['previous_treatments_ds'] + hold_out['previous_treatments_iui']+hold_out['previous_treatments_fet']+hold_out['previous_treatments_onc']+hold_out['previous_treatments_pgt']+hold_out['previous_treatments_ado']

data = data.drop(columns=[col for col in data.columns if col.startswith('current_pharmacy')])
drop_columns = ['mem_id_no_client', 'patient_id', 'onboard_date', 'target_date', 'events','days_since_last_callog']
X = data.drop(drop_columns, axis=1)  # Adjust 'target_column' to the name of your target column
y = data['events']

# Define the outer cross-validation loop
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# Define the inner cross-validation loop
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Define the hyperparameter grid

param_grid ={
    'n_estimators': [ 300],
    'learning_rate': [ 0.05],
    'lambda': [1],
'max_depth': [4],
     'learning_rate':  [0.05],
     'reg_alpha':  [1],
   'lambda':  [1],
    'gamma':  [0.1],
'min_child_weight': [ 5]
}

#'n_estimators': [ 200,300, 500],
# 'learning_rate': [0.01, 0.05, 0.1],
# 'lambda': [0.1, 0.5, 1]

#'max_depth': [3, 4, 5],
#'reg_alpha': [0.1, 1, 1.5],
#'gamma': [0.1, 0.5],
#'subsample':[0.5,0.8,1]
# Define the Xgboost classifier param_grid = {
#     'n_estimators': [ 200, 500],
#     'max_depth': [3, 4, 5],
#     'learning_rate': [ 0.1],
#     'reg_alpha': [0.1, 1, 1.5],
#     'reg_lambda': [ 1, 1.5],
#     'gamma': [ 0.1, 0.5],
#     'min_child_weight': [1, 3, 5],
clf = XGBClassifier(**param_grid)

# Perform nested cross-validation

classification_rep = pd.DataFrame()
inner_cv_results_df = pd.DataFrame()
outer_train_scores = []
outer_test_scores = []
loop_index =[]
outer_cv_outputs_df =pd.DataFrame()
for i, (train_index, test_index) in enumerate(outer_cv.split(X, y)):
    print(f"Fold {i}")
    print('Set difference of train and test indexes {}'.format(len(set(train_index)&set(test_index))))
    X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    print("run train_test ")
    # Perform inner cross-validation to find the best hyperparameters
    grid_search = GridSearchCV(clf, param_grid, cv=inner_cv, return_train_score=True,scoring='roc_auc')
    grid_search.fit(X_train, y_train)
    print("run fit ")
    # Get the best hyperparameters and score
    best_params_inner = grid_search.best_params_
    best_score_inner = grid_search.best_score_
    best_est = grid_search.best_estimator_
    inner_cv_results = grid_search.cv_results_
    # Convert the cv_results to a DataFrame
    inner_cv_results_df = inner_cv_results_df._append(pd.DataFrame(inner_cv_results), ignore_index=True)
    print("scores and params ")
    print(f"Fold {i} cv BEST SCORE SELECTED{best_score_inner}")
    print(f"Fold {i} cv BEST PARAMETTER SELECTED {best_params_inner}")
    # Train the Xgboost classifier with the best hyperparameters
    clf.set_params(**best_params_inner)
    clf.fit(X_train, y_train)
    y_test_pred = clf.predict_proba(X_test)[:, 1]
    print("fit with best parameter ")
    # Evaluate the Xgboost classifier on the test set
    train_roc_score = roc_auc_score(y_train, clf.predict_proba(X_train)[:, 1])
    test_roc_score = roc_auc_score(y_test, y_test_pred)
    outer_test_classif_report = classification_report(y_test, clf.predict(X_test),output_dict=True)
    outer_test_classif_df= pd.DataFrame(outer_test_classif_report).transpose()
    outer_test_classif_df.reset_index(inplace=True)
    # Append the best score and best hyperparameters to the lists
    # summarize the estimated performance of the model
    classification_rep =pd.concat([classification_rep,outer_test_classif_df])
    params = pd.DataFrame.from_dict(best_params_inner,orient='index').T
    # Create a DataFrame from the outer_train_scores, outer_test_scores, and best_scores_in lists
    t1 = pd.DataFrame(columns= ['Fold','Outer_train_score','Outer_Test_ROC','Best_validate_scores'])
    t1.loc[0,'Fold'] = i
    t1.loc[0,'Outer_train_score'] = train_roc_score
    t1.loc[0,'Outer_Test_ROC'] = test_roc_score
    t1.loc[0,'Best_validate_scores'] = best_score_inner
    t1=pd.concat([t1,params],axis=1)
    outer_cv_outputs_df= pd.concat([outer_cv_outputs_df, t1])

# Save the inner loop cv results to an Excel file
# Save the workbook
workbook.save('nested_cv_outputs.xlsx')


with pd.ExcelWriter('nested_cv_output6.xlsx', mode='w') as writer:
    inner_cv_results_df.to_excel(writer, sheet_name='Inner_roc_all', index=False)

# Save the outer loop cv results to sheet 2 of an Excel file
with pd.ExcelWriter('nested_cv_output6.xlsx', mode='a') as writer:
    outer_cv_outputs_df.to_excel(writer, sheet_name='Outer_roc_all', index=False)

with pd.ExcelWriter('nested_cv_output6.xlsx', mode='a') as writer:
    classification_rep.to_excel(writer, sheet_name='test_Classif_all', index=False)

#AGGREGATING PREVIOUS TREATMENT VARIABLE
X_hl = hold_out.drop(drop_columns, axis=1)  # Adjust 'target_column' to the name of your target column
y_hl = hold_out['events']

#Building best model from best paramenters
param_grid={'n_estimators': 300,
     'max_depth': 4,
     'learning_rate':  0.05,
     'reg_alpha':  1,
   'lambda':  1,
    'gamma':  0.1,
  'min_child_weight':5
}
clf_best1 = XGBClassifier()
clf_best1.set_params(**param_grid)
print(clf_best1)
clf_best1.fit(X,y)
y_hl_prob = clf_best1.predict_proba(X_hl)[:, 1]
x_train_prob= clf_best1.predict_proba(X)[:,1]
y_hl_p = clf_best1.predict(X_hl)
hl_score = roc_auc_score(y_hl,y_hl_prob)
train_score = roc_auc_score( y,x_train_prob)

print(train_score)
print(hl_score)
cohen_test= cohen_kappa_score(y_hl, y_hl_p)
print(cohen_test)
# Save the model
joblib.dump(clf_best, 'Xgboost_roc_tr82_ts80_v2.pkl')


# Feat importance using permutation  improtance
for feat, importance in zip(X.columns,clf_best.feature_importances_):
    print('{}, {}'.format(feat,importance))
# Calculate permutation importance
feature_importances = permutation_importance(model, X, y, n_repeats=10)
# Print feature importance
print('Feature Importance:')
for feature, importance in zip(X.columns, feature_importances['mean_importance']):
    print(f'{feature}: {importance}')

import matplotlib as plt
from shap import TreeExplainer
# Create a SHAP explainer
explainer = shap.TreeExplainer(clf_best)
# Calculate SHAP values
shap_values = explainer.shap_values(X)
shap_values.shape
# Create a SHAP chart
#shap.summary_plot(shap_values, X, plot_type='bar')
shap.summary_plot(shap_values,X)
plt.title('SHAP Chart')
plt.show()
#Model on features with just important features

#ref feature selection

from sklearn.feature_selection import RFE

# Create an RFE estimator
rfe = RFE(estimator=clf_best, n_features_to_select=25)  # Select top 10 features
rfe.fit(X, y)
selected_features = X.columns[rfe.support_]
print("Selected features:", selected_features)
X_selected = X[selected_features]


clf_best_new = XGBClassifier() # Use selected features to train a new XGBoost model
clf_best_new.set_params(**param_grid)
clf_best_new.fit(X_selected, y)

#Model performance
X_hl_selected = X_hl[selected_features]
y_hl_prob = clf_best_new.predict_proba(X_hl_selected)[:, 1]
x_train_prob= clf_best_new.predict_proba(X_selected)[:,1]

hl_score = roc_auc_score(y_hl,y_hl_prob)
train_score = roc_auc_score( y,x_train_prob)
print(train_score)
print(hl_score)


print(hold_out.groupby('events').count())
