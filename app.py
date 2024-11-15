# Streamlit ARIMA Forecasting and Machine Learning App
import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
st.image("https://upload.wikimedia.org/wikipedia/en/d/d3/BITS_Pilani-Logo.svg", width=300)
st.title("DISSERTATION")

# Load dataset
uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])
if uploaded_file is not None:
    data = pd.read_excel(uploaded_file)

    # Define list of months (Jan to Nov 2024)
    months_list = pd.date_range("2024-01-01", "2024-11-01", freq="MS").strftime('%b %Y').tolist()
    
    # Add a synthetic "Month" column if it doesn't exist
    if 'Month' not in data.columns:
        random_months = np.random.choice(months_list, size=len(data))
        data['Month'] = random_months
    data['Month'] = pd.to_datetime(data['Month'], format='%b %Y')
    data.set_index('Month', inplace=True)

    # Streamlit App Layout
    st.title("Forecasting and Classification App")

    # Creating Tabs
    tab1, tab2, tab3 = st.tabs(["ARIMA Forecasting", "Ensemble Model","Prediction"])

    # ARIMA Forecasting Tab
    with tab1:
        st.header("ARIMA Forecasting")
        if st.button("Show Dataset Info (ARIMA)"):
            st.write(f"Number of Rows: {data.shape[0]}")
            st.write(f"Number of Columns: {data.shape[1]}")
            st.write(data.info())
        if 'Tenure' in data.columns:
            monthly_data = data['Tenure'].resample('M').mean()
            model = ARIMA(monthly_data, order=(1, 1, 1))
            model_fit = model.fit()

            # Forecast the next 3 months
            forecast_steps = 3
            forecast = model_fit.forecast(steps=forecast_steps)
            forecast_dates = pd.date_range(monthly_data.index[-1], periods=forecast_steps + 1, freq='M')[1:]

            # Plot with Plotly
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=monthly_data.index, y=monthly_data, mode='lines+markers', name='Historical Data'))
            fig.add_trace(go.Scatter(x=forecast_dates, y=forecast, mode='lines+markers', name='Forecast', line=dict(dash='dash', color='red')))
            fig.add_vline(x=monthly_data.index[-1], line=dict(color='gray', dash='dash'), name='Forecast Start')

            fig.update_layout(
                title="ARIMA Forecast for Tenure",
                xaxis_title="Month",
                yaxis_title="Average Tenure",
                hovermode="x unified",
                legend=dict(x=0.02, y=0.98),
                template="plotly_white"
            )

            st.plotly_chart(fig)
        else:
            st.warning("Column 'Tenure' not found in the uploaded file.")

    # Classification Model Tab
    with tab2:
        st.header("Classification Model")
        
        # Select columns for classification
        columns_to_use = [
            'Professional Level', 'Gender', '2022 Rating', '2023 Rating',
            'Grade/Title', 'Job Family', 'Tenure', 'Work Status'
        ]

        # Data Cleaning
        df = data[columns_to_use].copy()
        df.dropna(inplace=True)

        # Target variable - 'Work Status'
        if 'Work Status' in df.columns:
            df['Work Status'] = df['Work Status'].apply(lambda x: 1 if x == 'Resigned' else 0)

            # Label Encoding for categorical variables
            label_encoder = LabelEncoder()
            categorical_columns = ['Professional Level', 'Gender', '2022 Rating', '2023 Rating', 'Grade/Title', 'Job Family']
            for col in categorical_columns:
                df[col] = label_encoder.fit_transform(df[col])

            # Separating target and features
            X = df.drop(columns=['Work Status'])
            y = df['Work Status']

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Feature Scaling
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Logistic Regression Model
            log_reg = LogisticRegression(class_weight='balanced', random_state=42)
            log_reg.fit(X_train, y_train)
            y_pred_log_reg = log_reg.predict(X_test)

            # Random Forest Classifier Model
            rf_clf = RandomForestClassifier(class_weight='balanced', random_state=42)
            rf_clf.fit(X_train, y_train)
            y_pred_rf = rf_clf.predict(X_test)

            # Bagging Classifier with Logistic Regression
            bagging_clf = BaggingClassifier(estimator=log_reg, n_estimators=10, random_state=42)
            bagging_clf.fit(X_train, y_train)
            y_pred_bagging = bagging_clf.predict(X_test)

            # Model Evaluation
            st.subheader("Logistic Regression Results")
            st.write("Accuracy:", accuracy_score(y_test, y_pred_log_reg))
            report_log_reg = classification_report(y_test, y_pred_log_reg, output_dict=True)
            report_log_reg_df = pd.DataFrame(report_log_reg).transpose() 
            report_log_reg_df = report_log_reg_df.round(2)  
            
            # Display as a table
            st.table(report_log_reg_df) 

            st.subheader("Random Forest Classifier Results")
            st.write("Accuracy:", accuracy_score(y_test, y_pred_rf))
            report_rf = classification_report(y_test, y_pred_rf, output_dict=True)  
            report_rf_df = pd.DataFrame(report_rf).transpose() 
            report_rf_df = report_rf_df.round(2)
            
            # Display as a table
            st.table(report_rf_df)

            st.subheader("Ensemble/bagging Results")
            st.write("Accuracy:", accuracy_score(y_test, y_pred_bagging))
            report_bagging = classification_report(y_test, y_pred_bagging, output_dict=True)
            report_bagging_df = pd.DataFrame(report_bagging).transpose()
            report_bagging_df = report_bagging_df.round(2)
            # Display as a table
            st.table(report_bagging_df)

        else:
            st.warning("Column 'Work Status' not found in the uploaded file.")
