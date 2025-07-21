import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd

# Load and preprocess the dataset
df = pd.read_csv("Salary_Data.csv")

# Basic preprocessing
df.dropna(inplace=True)
df['Education Level'].replace(["Bachelor's Degree", "Master's Degree", "phD"],
                              ["Bachelor's", "Master's", "PhD"], inplace=True)
df['Gender'] = df['Gender'].astype('category').cat.codes
education_mapping = {"High School": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3}
df['Education Level'] = df['Education Level'].map(education_mapping)

# Handle low-frequency job titles
job_title_count = df['Job Title'].value_counts()
low_freq_jobs = job_title_count[job_title_count <= 25].index
df['Job Title'] = df['Job Title'].apply(lambda x: 'Others' if x in low_freq_jobs else x)
df = pd.get_dummies(df, columns=['Job Title'], drop_first=True)

# Split features and target
X = df.drop('Salary', axis=1)
y = df['Salary']

# Train/test split and model training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
model = RandomForestRegressor(n_estimators=20)
model.fit(X_train, y_train)

# Save feature columns for consistent input
model_features = X.columns

# Streamlit UI
st.title("Salary Prediction App")

age = st.slider("Age", 18, 65, 30)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
education = st.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD"])
experience = st.slider("Years of Experience", 0, 40, 5)

# Handle job title selection
job_titles = [col for col in model_features if col.startswith("Job Title_")]
job = st.selectbox("Job Title", [j.replace("Job Title_", "") for j in job_titles])
job_dict = {f"Job Title_{j.replace('Job Title_', '')}": 1 if job == j.replace("Job Title_", "") else 0 for j in job_titles}

# Create input data dictionary
input_data = {
    'Age': age,
    'Gender': {"Male": 1, "Female": 0, "Other": 2}[gender],
    'Education Level': education_mapping[education],
    'Years of Experience': experience,
}
input_data.update(job_dict)

# Create DataFrame and align with training features
input_df = pd.DataFrame([input_data])

# Ensure all model features are present in input_df
for col in model_features:
    if col not in input_df.columns:
        input_df[col] = 0

# Reorder columns to match training data
input_df = input_df[model_features]

# Make prediction
prediction = model.predict(input_df)[0]

# Display result
st.success(f"Estimated Salary: ${prediction:,.2f}")
