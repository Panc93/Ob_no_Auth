import pandas as pd
from sklearn.feature_selection import mutual_info_classif
import joblib
import openpyxl

# Load the pickle file containing your dataset
data = joblib.load('your_data.pkl')  # Replace 'your_data.pkl' with the actual file path

# Load the column details file with variable types (e.g., numeric, ordinal, categorical)
column_details = pd.read_csv('column_details.csv')  # Replace 'column_details.csv' with the actual file path

# Assuming you have a binary target variable named 'events' in your data
target = data['events']

# Initialize a dictionary to store mutual information scores
mi_scores = {}

# Iterate through the columns in column_details
for index, row in column_details.iterrows():
    column_name = row['Column Name']
    data_type = row['Data Type']

    if data_type == 'numeric':
        # Numeric variable
        numeric_data = data[column_name]
        mi_score = mutual_info_classif(numeric_data.values.reshape(-1, 1), target, discrete_values=False)
        mi_scores[column_name] = mi_score[0]

    elif data_type == 'ordinal':
        # Ordinal variable (no change)
        ordinal_data = data[column_name]
        mi_score = mutual_info_classif(ordinal_data.values.reshape(-1, 1), target, discrete_values=True)
        mi_scores[column_name] = mi_score[0]

    elif data_type == 'categorical':
        # Categorical variable (one-hot encoding)
        categorical_data = data[column_name]
        categorical_data_encoded = pd.get_dummies(categorical_data, prefix=column_name)
        mi_score = mutual_info_classif(categorical_data_encoded, target, discrete_values=True)
        mi_scores[column_name] = sum(mi_score)

    elif data_type == 'binary':
        # Binary variable (assuming no encoding required)
        binary_data = data[column_name]
        mi_score = mutual_info_classif(binary_data.values.reshape(-1, 1), target, discrete_values=True)
        mi_scores[column_name] = mi_score[0]

# Print or use mi_scores as needed
print("Mutual Information Scores:")
for column, score in mi_scores.items():
    print(f"{column}: {score}")

# Create a DataFrame for the mutual information scores
mi_scores_df = pd.DataFrame(list(mi_scores.items()), columns=['Variable', 'Mutual Information Score'])

# Save the output to Excel and pickle files
mi_scores_df.to_excel('mutual_information_scores.xlsx', index=False, engine='openpyxl')
mi_scores_df.to_pickle('mutual_information_scores.pkl')
