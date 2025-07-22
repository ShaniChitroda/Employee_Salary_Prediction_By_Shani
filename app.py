import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("best_model.pkl")

# Define the exact features used during training
required_columns = ['age', 'education', 'occupation', 'hours-per-week']

st.set_page_config(page_title="Employee Salary Classification", page_icon="💼", layout="centered")

st.title("💼 Employee Salary Classification App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")

# Sidebar for user input
st.sidebar.header("Input Employee Details")

# Input fields
age = st.sidebar.slider("Age", 18, 65, 30)
education = st.sidebar.selectbox("Education Level", [
    "Bachelors", "Masters", "PhD", "HS-grad", "Assoc", "Some-college"
])
occupation = st.sidebar.selectbox("Job Role", [
    "Tech-support", "Craft-repair", "Other-service", "Sales",
    "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct",
    "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv",
    "Protective-serv", "Armed-Forces"
])
hours_per_week = st.sidebar.slider("Hours per week", 1, 80, 40)

# Build single prediction input DataFrame
input_df = pd.DataFrame({
    'age': [age],
    'education': [education],
    'occupation': [occupation],
    'hours-per-week': [hours_per_week]
})

st.write("### 🔎 Input Data")
st.write(input_df)

# Predict on button click
if st.button("Predict Salary Class"):
    prediction = model.predict(input_df)
    st.success(f"✅ Prediction: {prediction[0]}")

# Batch prediction
st.markdown("---")
st.markdown("#### 📂 Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type="csv")

if uploaded_file is not None:
    try:
        batch_data = pd.read_csv(uploaded_file)
        st.write("📄 Uploaded Data Preview:", batch_data.head())

        # Filter to required columns and handle missing ones
        missing_cols = [col for col in required_columns if col not in batch_data.columns]
        if missing_cols:
            st.error(f"❌ Missing required columns in uploaded CSV: {missing_cols}")
        else:
            # Optional: Remove rows with missing or unknown values
            clean_data = batch_data[required_columns].replace("?", pd.NA).dropna()
            predictions = model.predict(clean_data)
            batch_data = batch_data.loc[clean_data.index]  # Keep aligned with predictions
            batch_data['PredictedClass'] = predictions
            st.write("✅ Predictions:")
            st.dataframe(batch_data)

            # Download predictions
            csv = batch_data.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions CSV", csv, file_name='predicted_classes.csv', mime='text/csv')
    except Exception as e:
        st.error(f"⚠️ Error processing file: {e}")
