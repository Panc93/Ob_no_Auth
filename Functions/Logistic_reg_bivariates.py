# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 16:16:49 2023

@author: ParnikaPancholi
"""

import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc

# Load your dataset (replace 'your_data.csv' with the path to your dataset)
src = r'C:\Users\ParnikaPancholi\PycharmProjects\Ob_no_Auth\Data\Input'
indsn = 'Ob_Noic_23_v2'
hl= 'Ob_Noic_23_v2_hl'
data = pd.read_pickle(r"{}\{}.pkl".format(str(src), str(indsn)))
hold_out = pd.read_pickle(r"{}\{}.pkl".format(str(src), str(hl)))
drop_columns = ['mem_id_no_client', 'patient_id', 'onboard_date', 'target_date', 'events']
# Define your features (X) and target (y)
X = data.drop(drop_columns, axis=1)  # Adjust 'target_column' to the name of your target column
y = data['events']
X_hl =hold_out.drop(drop_columns, axis=1)
y_hl =hold_out['events']
hold_out.shape()
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=42)

# Create a Logistic Regression model
# Create a Decision Tree model
model = DecisionTreeClassifier()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)
# Convert the predicted labels to predicted probabilities.
y_pred_proba = model.predict_proba(X_test)
y_hl_pred = model.predict(X_hl)
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", classification_rep)

accuracy = accuracy_score(y_hl, y_hl_pred)
classification_rep = classification_report(y_hl, y_hl_pred)

print("Accuracy holdout:", accuracy)
print("Classification Report hold_out:\n", classification_rep)
def calculate_roc_curve_and_metric(y_true, y_pred_proba):
    """Calculates the ROC curve and metric of a classification model.

  Args:
    y_true: A NumPy array containing the true labels.
    y_pred_proba: A NumPy array containing the predicted probabilities.

  Returns:
    A tuple containing the ROC curve and metric.
  """

    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba[:, 1])
    roc_auc_score = auc(fpr, tpr)

    return fpr, tpr, roc_auc_score

y_pred_proba = model.predict_proba(X_train)
y_test_pred_proba = model.predict_proba(X_test)
y_hl_pred_proba = model.predict_proba(X_hl)
# Calculate the ROC curve and metric.
fpr, tpr, roc_auc_scoret = calculate_roc_curve_and_metric(y_train, y_pred_proba)
fpr, tpr, roc_auc_scorete = calculate_roc_curve_and_metric(y_test, y_test_pred_proba)
fpr, tpr, roc_auc_scorehl = calculate_roc_curve_and_metric(y_hl, y_hl_pred_proba)
# Plot the ROC curve.

import matplotlib.pyplot as plt

plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

# Print the ROC AUC score.
print(f'ROC AUC score: {roc_auc_scoret}')
print(f'ROC AUC score: {roc_auc_scorete}')
print(f'ROC AUC score: {roc_auc_scorehl}')

for feat, importance in zip(X.columns,model.feature_importances_):
    print('{}, {}'.format(feat,importance))

fig = plt.figure(figsize=(24, 25))
_ = tree.plot_tree(model, feature_names=X_train.columns, class_names=['0', '1'], filled=True)