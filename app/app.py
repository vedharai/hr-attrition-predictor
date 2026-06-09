import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Loading data and model
df = pd.read_csv('/Users/vedharai/hr-attrition-predictor/data/hr_clean.csv')
model = joblib.load('/Users/vedharai/hr-attrition-predictor/app/attrition_model.pkl')
#scaler = joblib.load('/Users/vedharai/hr-attrition-predictor/app/scaler.pkl')

# App title
st.title("HR Attrition Predictor")
st.write("Analyse employee attrition patterns and predict which employees are at risk of leaving.")

st.header("Attrition Overview")

# Attrition count
attrition_counts = df['Attrition'].value_counts().rename({0: 'Stayed', 1: 'Left'})
st.bar_chart(attrition_counts)

# Attrition by department - using original data
st.subheader("Key Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Total Employees", len(df))
col2.metric("Attrition Rate", f"{df['Attrition'].mean()*100:.1f}%")
col3.metric("Employees at Risk", df['Attrition'].sum())

st.header("What Drives Attrition?")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
import numpy as np

X = df.drop(columns=['Attrition'])
y = df['Attrition']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
undersampler = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = undersampler.fit_resample(X_train, y_train)
rf = RandomForestClassifier(max_depth=3, min_samples_leaf=10, n_estimators=100, class_weight='balanced', random_state=42)
rf.fit(X_resampled, y_resampled)

importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': rf.feature_importances_})
top_features = importance_df.sort_values(by='Importance', ascending=False).head(15)

fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(x='Importance', y='Feature', data=top_features, ax=ax)
ax.set_title('Top 15 Features Driving Attrition')
st.pyplot(fig)

st.header("Predict Attrition Risk")
st.write("Enter employee details below.")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 60, 36, key="age")
    monthly_income = st.slider("Monthly Income", 1000, 20000, 6500, key="income")
    total_working_years = st.slider("Total Working Years", 0, 40, 10, key="twy")
    years_at_company = st.slider("Years at Company", 0, 40, 7, key="yac")

with col2:
    job_level = st.slider("Job Level (1-5)", 1, 5, 2, key="jl")
    stock_option = st.slider("Stock Option Level (0-3)", 0, 3, 1, key="so")
    job_satisfaction = st.slider("Job Satisfaction (1-4)", 1, 4, 3, key="js")
    overtime = st.selectbox("Works Overtime?", ["No", "Yes"], key="ot")

if st.button("Predict"):
    input_data = pd.DataFrame([df.drop(columns=['Attrition']).mean()])
    input_data['Age'] = age
    input_data['MonthlyIncome'] = monthly_income
    input_data['TotalWorkingYears'] = total_working_years
    input_data['YearsAtCompany'] = years_at_company
    input_data['JobLevel'] = job_level
    input_data['StockOptionLevel'] = stock_option
    input_data['JobSatisfaction'] = job_satisfaction
    input_data['OverTime_Yes'] = 1 if overtime == "Yes" else 0

    probability = model.predict_proba(input_data.values)[0][1]
    prediction = 1 if probability >= 0.5 else 0

    if prediction == 1:
        st.error(f"High attrition risk — {probability*100:.1f}% chance of leaving")
    else:
        st.success(f"Low attrition risk — {probability*100:.1f}% chance of leaving")