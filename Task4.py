import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
df = pd.read_csv(r"C:\Users\mariu\OneDrive\Desktop\work\Intership\Disease_symptom_and_patient_profile_dataset.csv")
print("Dataset Preview:")
print(df.head())
print("Column Names:")
print(df.columns)

# Data Preprocessing
# Encode categorical variables
label_encoders = {}
for column in ['Disease', 'Fever', 'Cough', 'Fatigue', 'Difficulty Breathing', 'Gender', 'Blood Pressure', 'Cholesterol Level', 'Outcome Variable']:
    encoder = LabelEncoder()
    df[column] = encoder.fit_transform(df[column])
    label_encoders[column] = encoder

# Verify encoded data
print("\nEncoded DataFrame:")
print(df.head())

# Split the data into features and target
X = df.drop('Outcome Variable', axis=1)
y = df['Outcome Variable']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f'\nModel Evaluation:')
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')

# Making Predictions for New Data
def make_prediction(new_data):
    # Encode new data
    encoded_data = {column: label_encoders[column].transform([value])[0] if column in label_encoders else value 
                    for column, value in new_data.items()}
    new_data_df = pd.DataFrame([encoded_data], columns=X.columns)
    prediction = model.predict(new_data_df)
    return label_encoders['Outcome Variable'].inverse_transform(prediction)[0]

# Example Predictions
new_data_1 = {
    'Disease': 'Influenza',
    'Fever': 'Yes',
    'Cough': 'No',
    'Fatigue': 'Yes',
    'Difficulty Breathing': 'Yes',
    'Age': 20,
    'Gender': 'Female',
    'Blood Pressure': 'Low',
    'Cholesterol Level': 'Normal'
}
print(f'Prediction for new data 1: {make_prediction(new_data_1)}')

new_data_2 = {
    'Disease': 'Common Cold',
    'Fever': 'No',
    'Cough': 'Yes',
    'Fatigue': 'Yes',
    'Difficulty Breathing': 'No',
    'Age': 25,
    'Gender': 'Female',
    'Blood Pressure': 'Normal',
    'Cholesterol Level': 'Normal'
}
print(f'Prediction for new data 2: {make_prediction(new_data_2)}')