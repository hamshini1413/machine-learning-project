import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load models and scalers
placement_model = joblib.load('placement_model.joblib')
salary_model = joblib.load('salary_model.joblib')
scaler = joblib.load('scaler.joblib')

# Create mappings for categorical variables
branch_mapping = {'ECE': 0, 'Chemical': 1, 'EE': 2, 'CE': 3, 'CSE': 4, 'IT': 5, 'ME': 6}
tier_mapping = {"Tier-1": 0, "Tier-2": 1, "Tier-3": 2}

st.set_page_config(page_title="Student Placement Predictor", layout="wide")

st.title("🎓 Student Placement & Salary Predictor")
st.write("Predict placement status and expected salary for students")

# Sidebar for navigation
page = st.sidebar.selectbox("Select Page", ["Predict Placement", "Batch Prediction"])

if page == "Predict Placement":
    st.header("Individual Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        branch = st.selectbox("Branch", ["ECE", "Chemical", "EE", "CE", "CSE", "IT", "ME"])
        college_tier = st.selectbox("College Tier", ["Tier-1", "Tier-2", "Tier-3"])
        cgpa = st.slider("CGPA", 0.0, 10.0, 7.5)
        backlogs = st.number_input("Backlogs", 0, 10, 0)
        coding_skills = st.slider("Coding Skills", 0.0, 10.0, 6.0)
        dsa_score = st.slider("DSA Score", 0.0, 10.0, 7.0)
    
    with col2:
        aptitude_score = st.slider("Aptitude Score", 0.0, 100.0, 70.0)
        communication_skills = st.slider("Communication Skills", 0.0, 10.0, 6.0)
        ml_knowledge = st.slider("ML Knowledge", 0.0, 10.0, 5.0)
        system_design = st.slider("System Design", 0.0, 10.0, 5.0)
        internships = st.number_input("Internships", 0, 10, 1)
        projects_count = st.number_input("Projects Count", 0, 20, 3)
        certifications = st.number_input("Certifications", 0, 10, 2)
        hackathons = st.number_input("Hackathons", 0, 10, 1)
        open_source = st.number_input("Open Source Contributions", 0, 10, 1)
        extracurriculars = st.number_input("Extracurriculars", 0, 5, 1)
    
    if st.button("Predict", key="predict_button"):
        # Encode categorical variables
        branch_encoded = branch_mapping[branch]
        college_tier_encoded = tier_mapping[college_tier]
        
        # Create input array
        input_data = np.array([
            branch_encoded,
            college_tier_encoded,
            cgpa,
            backlogs,
            coding_skills,
            dsa_score,
            aptitude_score,
            communication_skills,
            ml_knowledge,
            system_design,
            internships,
            projects_count,
            certifications,
            hackathons,
            open_source,
            extracurriculars
        ]).reshape(1, -1)
        
        # Scale the input
        input_scaled = scaler.transform(input_data)
        
        # Predict placement status
        placement_pred = placement_model.predict(input_scaled)[0]
        
        st.success(f"Placement Status: {'🎉 Placed' if placement_pred == 1 else '❌ Not Placed'}")
        
        if placement_pred == 1:
            salary_pred = salary_model.predict(input_scaled)[0]
            st.info(f"Expected Salary: ₹ {salary_pred:.2f} LPA")

elif page == "Batch Prediction":
    st.header("Batch Prediction from CSV")
    
    uploaded_file = st.file_uploader("Upload CSV file", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded file:")
        st.write(df.head())
        
        if st.button("Process Predictions"):
            # Process each row
            predictions = []
            salaries = []
            
            for idx, row in df.iterrows():
                # Encode categorical variables
                branch_encoded = branch_mapping[row['branch']]
                college_tier_encoded = tier_mapping[row['college_tier']]
                
                # Create input array
                input_data = np.array([
                    branch_encoded,
                    college_tier_encoded,
                    row['cgpa'],
                    row['backlogs'],
                    row['coding_skills'],
                    row['dsa_score'],
                    row['aptitude_score'],
                    row['communication_skills'],
                    row['ml_knowledge'],
                    row['system_design'],
                    row['internships'],
                    row['projects_count'],
                    row['certifications'],
                    row['hackathons'],
                    row['open_source_contributions'],
                    row['extracurriculars']
                ]).reshape(1, -1)
                
                # Scale the input
                input_scaled = scaler.transform(input_data)
                
                # Predict placement status
                placement_pred = placement_model.predict(input_scaled)[0]
                predictions.append(placement_pred)
                
                # Predict salary if placed
                if placement_pred == 1:
                    salary_pred = salary_model.predict(input_scaled)[0]
                    salaries.append(salary_pred)
                else:
                    salaries.append(None)
            
            # Add predictions to dataframe
            df['Predicted_Placement'] = predictions
            df['Predicted_Placement_Status'] = df['Predicted_Placement'].apply(lambda x: 'Placed' if x == 1 else 'Not Placed')
            df['Predicted_Salary_LPA'] = salaries
            
            st.write("Predictions:")
            st.write(df[['branch', 'college_tier', 'cgpa', 'Predicted_Placement_Status', 'Predicted_Salary_LPA']])
            
            # Download results as CSV
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv"
            )

st.sidebar.info("App created for student placement prediction")
