import streamlit as st
import pandas as pd
import numpy as np
import joblib

@st.cache_resource
def load_models():
    try:
        model = joblib.load('models/salary_prediction_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        encoder = joblib.load('models/encoder.pkl')
        return model, scaler, encoder
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
        return None, None, None

def preprocess_input(job_title,experience_level,employment_type,company_location,company_size,employee_residence,remote_ratio,education_required,years_experience,industry, scaler, encoder):
    if scaler is None:
        raise ValueError("Scaler not loaded properly")

    input_data = pd.DataFrame({
        'job_title': [job_title],
        'experience_level': [experience_level],
        'employment_type': [employment_type],
        'company_location': [company_location],
        'company_size': [company_size],
        'employee_residence': [employee_residence],
        'remote_ratio': [remote_ratio],
        'education_required': [education_required],
        'years_experience': [years_experience],
        'industry': [industry]
    })
    # Create an input dataframe with all 10 features that match your original model

    # Manual mapping for experience level (match your original model)
    exp_mapping = {'EN': 0, 'MI': 2, 'SE': 1, 'EX': 3}
    input_data['experience_level'] = input_data['experience_level'].map(exp_mapping)

    # Apply ordinal encoding for categorical variables
    categorical_cols = ['job_title', 'experience_level', 'employment_type','company_location','company_size',
                        'employee_residence', 'education_required', 'industry']
    
    # Fit and transform (note: in production, you should save the fitted encoder)
    input_data[categorical_cols] = encoder.fit_transform(input_data[categorical_cols])

    # Apply standard scaling
    input_scaled = scaler.transform(input_data)
    return input_scaled


def main():
    st.set_page_config(
        page_title="Salary Prediction App",
        page_icon="💰",
        layout="wide"
    )

    st.title("💰 AI Job Salary Prediction")
    st.markdown("Predict your salary based on job characteristics using machine learning!")

    # Load model and preprocessors
    model, scaler, encoder = load_models()

    if model is None or scaler is None or encoder is None:
        st.error("Failed to load models. Please ensure model files are available.")
        return

    # Create two columns for a better layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Job Information")

        job_title = st.selectbox(
            "Job Title",
            options=['Machine Learning Researcher','AI Software Engineer','Autonomous Systems Engineer',
                     'Machine Learning Engineer','AI Architect','Head of AI','NLP Engineer','Robotics Engineer',
                     'Data Analyst','AI Research Scientist','Data Engineer','AI Product Manager','Research Scientist',
                     'Principal Data Scientist','AI Specialist','ML Ops Engineer','Data Scientist','Computer Vision Engineer',
                     'Deep Learning Engineer','AI Consultant']
        )

        experience_level = st.selectbox(
            "Experience Level",
            options=["EN", "MI", "SE", "EX"],
            format_func=lambda x: {"EN": "Entry-level", "MI": "Mid-level",
                                  "SE": "Senior-level", "EX": "Executive-level"}[x]
        )
        
        years_experience = st.number_input(
            "Years of Experience",
            min_value=0,
            max_value=50,
            value=5,
            step=1,
            help="Enter your total years of professional experience"
        )

        employment_type = st.selectbox(
            "Employment Type",
            options=["FT", "PT", "CT", "FL"],
            format_func=lambda x: {"FT": "Full-time", "PT": "Part-time",
                                  "CT": "Contract", "FL": "Freelance"}[x]
        )

        work_setting = st.selectbox(
            "Work Setting",
            options=[0, 50, 100],
            format_func=lambda x: {0: "In-person", 50: "Hybrid", 100: "Remote"}[x]
        )
        

    with col2:
        st.subheader("Company Information")

        company_location = st.selectbox(
            "Company Location",
            options=["US", "GB", "CA", "DE", "FR", "IN", "NL", "ES", "AU", "BR", "JP", "Other"]
        )

        company_size = st.selectbox(
            "Company Size",
            options=["S", "M", "L"],
            format_func=lambda x: {"S": "Small (< 50 employees)",
                                  "M": "Medium (50-250 employees)",
                                  "L": "Large (> 250 employees)"}[x]
        )

        employee_residence = st.selectbox(
            "Employee Residence",
            options=["US", "GB", "CA", "DE", "FR", "IN", "NL", "ES", "AU", "BR", "JP", "Other"]
        )
        
        education_required = st.selectbox(
            "Education Required",
            options=["Bachelor's", "Master's", "PhD", "None"]
        )

        industry = st.selectbox(
            "Industry",
            options=["Technology", "Healthcare", "Finance", "Manufacturing",
                     "Retail", "Education", "Other"]
        )
    
    # Prediction button
    if st.button("Predict Salary", type="primary"):
        try:
            # Preprocess input
            input_processed = preprocess_input(
                job_title, experience_level, employment_type, company_location,
                company_size, employee_residence, work_setting, education_required,
                years_experience, industry, scaler, encoder
            )

            # Make prediction
            log_prediction = model.predict(input_processed)[0]
            salary_prediction = np.expm1(log_prediction)

            # Display results
            st.success("Prediction Complete!")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Predicted Salary", f"${salary_prediction:,.0f}")

            with col2:
                st.metric("Monthly Salary", f"${salary_prediction/12:,.0f}")

            with col3:
                st.metric("Hourly Rate", f"${salary_prediction/(52*40):,.0f}")

            # Additional information
            st.info(f"💡 This prediction is based on machine learning models trained on AI job salary data.")

        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

    # Footer
    st.markdown("---")
    st.markdown("*Predictions are estimates based on historical data and may not reflect actual salaries!!*")

if __name__ == "__main__":
    main()