import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib

st.set_page_config(page_title="Bank Churn Intelligence System", page_icon="🏦", layout="wide")

@st.cache_resource
def load_models():
    try:
        model = joblib.load('churn_model.pkl')
        scaler = joblib.load('scaler.pkl')
        features = joblib.load('feature_names.pkl')
        return model, scaler, features
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

st.title("🏦 Predictive Churn Intelligence System")
st.markdown("*Unified Mentor Pvt Ltd | ECB Compliant*")
st.markdown("---")

model, scaler, features = load_models()
if model is None:
    st.stop()

# Sidebar inputs
st.sidebar.header("Customer Risk Assessment")
credit_score = st.sidebar.slider("Credit Score", 300, 850, 650)
age = st.sidebar.slider("Age", 18, 100, 35)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
geography = st.sidebar.selectbox("Geography", ["France", "Spain", "Germany"])
tenure = st.sidebar.slider("Tenure (Years)", 0, 10, 5)
balance = st.sidebar.number_input("Balance ($)", 0, 250000, 50000)
num_products = st.sidebar.selectbox("Number of Products", [1, 2, 3, 4])
has_cr_card = st.sidebar.selectbox("Has Credit Card?", ["Yes", "No"])
is_active = st.sidebar.selectbox("Active Member?", ["Yes", "No"])
estimated_salary = st.sidebar.number_input("Estimated Salary ($)", 0, 200000, 50000)

# Convert to numeric
has_cr_card = 1 if has_cr_card == "Yes" else 0
is_active = 1 if is_active == "Yes" else 0

# Derived features (must match training)
balance_salary_ratio = balance / (estimated_salary + 1)
product_density = num_products / 4
engagement_product = is_active * num_products
age_tenure = age * tenure
wealth_index = balance + estimated_salary
high_value = 1 if balance > 50000 else 0
senior = 1 if age >= 60 else 0
zero_balance = 1 if balance == 0 else 0

# Create input DataFrame with exact feature names used in training
input_data = pd.DataFrame([{
    'CreditScore': credit_score,
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_products,
    'HasCrCard': has_cr_card,
    'IsActiveMember': is_active,
    'EstimatedSalary': estimated_salary,
    'Balance_Salary_Ratio': balance_salary_ratio,
    'Product_Density': product_density,
    'Engagement_Product': engagement_product,
    'Age_Tenure': age_tenure,
    'Wealth_Index': wealth_index,
    'High_Value': high_value,
    'Senior': senior,
    'Zero_Balance': zero_balance,
    'Geography_Germany': 1 if geography == "Germany" else 0,
    'Geography_Spain': 1 if geography == "Spain" else 0,
    'Gender_Male': 1 if gender == "Male" else 0
}])

# Ensure all training features are present
for col in features:
    if col not in input_data.columns:
        input_data[col] = 0
input_data = input_data[features]

# Scale and predict
input_scaled = scaler.transform(input_data)
churn_prob = model.predict_proba(input_scaled)[0, 1]

# Display metrics
col1, col2, col3 = st.columns(3)
col1.metric("Churn Probability", f"{churn_prob:.1%}")
risk = "High" if churn_prob > 0.5 else "Medium" if churn_prob > 0.3 else "Low"
col2.metric("Risk Level", risk)
rec = "Immediate action" if churn_prob > 0.5 else "Monitor" if churn_prob > 0.3 else "Normal"
col3.metric("Recommendation", rec)

# Gauge chart
fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=churn_prob * 100,
    title={"text": "Risk Score"},
    gauge={
        "axis": {"range": [0, 100]},
        "steps": [
            {"range": [0, 30], "color": "green"},
            {"range": [30, 70], "color": "yellow"},
            {"range": [70, 100], "color": "red"}
        ]
    }
))
fig.update_layout(height=300)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("*Built for Unified Mentor Private Limited*")
