import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
data = pd.read_excel("input_data.xlsx")

# Data Cleaning and Preprocessing
columns_to_use = [
    'Professional Level', 'Gender', '2022 Rating', '2023 Rating',
    'Grade/Title', 'Job Family', 'Tenure', 'Work Status'
]
df = data[columns_to_use].copy()
df.dropna(inplace=True)

# Encode 'Work Status' as the target variable
df['Work Status'] = df['Work Status'].apply(lambda x: 1 if x == 'Resigned' else 0)

# Label Encoding for categorical variables
label_encoder_dict = {}
categorical_columns = ['Professional Level', 'Gender', '2022 Rating', '2023 Rating']

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])  # Fit and transform the data
    label_encoder_dict[col] = le  # Store encoder for future use

# Separating features and target variable
X = df.drop(columns=['Work Status'])
y = df['Work Status']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Training (Random Forest Classifier)
rf_model = RandomForestClassifier(class_weight='balanced', random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Streamlit UI
st.title("Employee Resignation Prediction")

# Create tabs for the app
tab1, tab2, tab3 = st.tabs(["ARIMA Forecast", "Model Evaluation", "Prediction"])

# Tab 1: ARIMA Forecast (Placeholder for ARIMA code)
with tab1:
    st.header("ARIMA Forecast")
    st.write("ARIMA model forecast coming soon...")

# Tab 2: Model Evaluation
with tab2:
    st.header("Model Evaluation")
    
    # Logistic Regression Evaluation
    st.subheader("Random Forest Classifier Results")
    y_pred_rf = rf_model.predict(X_test_scaled)
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    st.write(f"Accuracy: {accuracy_rf:.2f}")
    
    # Classification Report
    classification_report_rf = classification_report(y_test, y_pred_rf, output_dict=True)
    st.table(pd.DataFrame(classification_report_rf).transpose())

# Tab 3: Prediction
with tab3:
    st.header("Prediction: Will an Employee Resign?")
    
    # Input form
    st.subheader("Enter Employee Details")
    professional_level = st.selectbox("Professional Level", label_encoder_dict['Professional Level'].classes_)
    gender = st.radio("Gender", label_encoder_dict['Gender'].classes_)
    tenure = st.number_input("Tenure (in months)", min_value=0)
    rating_choices = ['No Review', 'Off Track', 'Effective', 'Outstanding']
    rating_2022 = st.selectbox("2022 Rating", rating_choices)
    rating_2023 = st.selectbox("2023 Rating", rating_choices)
    
    # Prediction button
    if st.button("Predict Resignation"):
        # Transform inputs using encoders
        professional_level_encoded = label_encoder_dict['Professional Level'].transform([professional_level])[0]
        gender_encoded = label_encoder_dict['Gender'].transform([gender])[0]
        rating_2022_encoded = label_encoder_dict['2022 Rating'].transform([rating_2022])[0]
        rating_2023_encoded = label_encoder_dict['2023 Rating'].transform([rating_2023])[0]
        
        # Create DataFrame for prediction
        input_data = pd.DataFrame({
            'Professional Level': [professional_level_encoded],
            'Gender': [gender_encoded],
            'Tenure': [tenure],
            '2022 Rating': [rating_2022_encoded],
            '2023 Rating': [rating_2023_encoded]
        })
        
        # Standardize the input data
        input_data_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = rf_model.predict(input_data_scaled)[0]
        result = "Will Resign" if prediction == 1 else "Will Not Resign"
        st.success(f"Prediction: {result}")
