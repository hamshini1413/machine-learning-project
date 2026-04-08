Student Placement & Salary Predictor

A Machine Learning + Streamlit web application that predicts whether a student will get placed and estimates their expected salary based on academic performance, skills, and activities.

Project Overview

This project uses machine learning models to analyze student data such as **CGPA, coding skills, internships, projects, communication skills, and more** to predict:

 Placement Status (Placed / Not Placed)
 Expected Salary (LPA)

The application provides:

* Individual Prediction – Enter student details manually.
* Batch Prediction – Upload a CSV file to predict placement results for multiple students.

The interactive interface is built using Streamlit.

 Features

* Predict student placement status
* Estimate expected salary
* Simple Streamlit UI
* Batch predictions using CSV upload
* Download prediction results as CSV

 Machine Learning Models

Two trained models are used:

1. Placement Model

    Predicts if a student will be placed or not.

2. Salary Model

    Predicts expected salary for placed students.

Data is scaled using StandardScaler before prediction.

Project Structure


project-folder
│
├── streamlit_app.py
├── placement_model.joblib
├── salary_model.joblib
├── scaler.joblib
├── placement.ipynb
├── requirements.txt
└── README.md


Main application file: 
 Input Features

The model uses the following features:

* Branch
* College Tier
* CGPA
* Backlogs
* Coding Skills
* DSA Score
* Aptitude Score
* Communication Skills
* ML Knowledge
* System Design
* Internships
* Projects Count
* Certifications
* Hackathons
* Open Source Contributions
* Extracurricular Activities

 How to Run the Project

 1.Clone the repository


git clone https://github.com/hamshini1413/machine-learning-project
cd student-placement-predictor


2. Install dependencies

pip install -r requirements.txt


3. Run the Streamlit app


streamlit run streamlit_app.py



 Batch Prediction Format

The uploaded CSV must contain columns like:

branch, college_tier, cgpa, backlogs, coding_skills, dsa_score,
aptitude_score, communication_skills, ml_knowledge, system_design,
internships, projects_count, certifications, hackathons,
open_source_contributions, extracurriculars

 Technologies Used

* Python
* Streamlit
* Scikit-learn
* Pandas
* NumPy
* Joblib

 Future Improvements

* Add model accuracy metrics in UI
* Deploy the app on **Streamlit Cloud**
* Improve model performance with more data
* Add visualization dashboards

 Author

Developed as a Machine Learning project for predicting student placement outcomes.
