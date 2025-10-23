import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Page configuration
st.set_page_config(
    page_title="Heart Disease Prediction System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        padding-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .positive {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
    }
    .negative {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
    }
    .info-box {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load the trained model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('best_heart_model.pkl')
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'best_heart_model.pkl' is in the same directory.")
        return None

# Function to preprocess input data
def preprocess_input(data):
    # Create a DataFrame with the input data
    df = pd.DataFrame([data])
    
    # One-hot encode categorical columns
    categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    df = pd.get_dummies(df, columns=categorical_cols)
    
    # Standardize numerical columns
    num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    scaler = StandardScaler()
    
    # Load the original dataset to fit the scaler (or save the scaler during training)
    try:
        original_data = pd.read_csv('dataset.csv')
        scaler.fit(original_data[num_cols])
        df[num_cols] = scaler.transform(df[num_cols])
    except:
        st.warning("Could not load original dataset for scaling. Using raw values.")
    
    return df

# Main application
def main():
    # Header
    st.markdown('<div class="main-header">Heart Disease Prediction System</div>', unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    if model is None:
        return
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Prediction", "About the Model", "Feature Information"])
    
    if page == "Prediction":
        show_prediction_page(model)
    elif page == "About the Model":
        show_about_page()
    else:
        show_feature_info_page()

def show_prediction_page(model):
    st.header("Patient Data Input")
    
    # Create two columns for input fields
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Basic Information")
        age = st.number_input("Age", min_value=1, max_value=120, value=50, help="Patient's age in years")
        sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        
        st.subheader("Blood Pressure & Cholesterol")
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
        chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], 
                          format_func=lambda x: "No" if x == 0 else "Yes")
        
        st.subheader("Heart Rate")
        thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
    
    with col2:
        st.subheader("Clinical Findings")
        cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3],
                         format_func=lambda x: ["Typical Angina", "Atypical Angina", 
                                               "Non-anginal Pain", "Asymptomatic"][x])
        
        restecg = st.selectbox("Resting ECG Results", options=[0, 1, 2],
                              format_func=lambda x: ["Normal", "ST-T Wave Abnormality", 
                                                    "Left Ventricular Hypertrophy"][x])
        
        exang = st.selectbox("Exercise Induced Angina", options=[0, 1],
                           format_func=lambda x: "No" if x == 0 else "Yes")
        
        oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, 
                                 value=0.0, step=0.1)
        
        slope = st.selectbox("Slope of Peak Exercise ST Segment", options=[0, 1, 2],
                           format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x])
        
        ca = st.selectbox("Number of Major Vessels (0-4)", options=[0, 1, 2, 3, 4])
        
        thal = st.selectbox("Thalassemia", options=[0, 1, 2, 3],
                          format_func=lambda x: ["Normal", "Fixed Defect", 
                                                "Reversible Defect", "Unknown"][x])
    
    # Prediction button
    if st.button("Predict Heart Disease Risk", type="primary"):
        # Prepare input data
        input_data = {
            'age': age,
            'sex': sex,
            'cp': cp,
            'trestbps': trestbps,
            'chol': chol,
            'fbs': fbs,
            'restecg': restecg,
            'thalach': thalach,
            'exang': exang,
            'oldpeak': oldpeak,
            'slope': slope,
            'ca': ca,
            'thal': thal
        }
        
        # Preprocess and predict
        try:
            processed_data = preprocess_input(input_data)
            
            # Get the expected features from the model
            try:
                expected_features = joblib.load('model_features.pkl')
                # Reindex to match training features
                processed_data = processed_data.reindex(columns=expected_features, fill_value=0)
            except:
                st.warning("Could not load feature list. Proceeding with available features.")
            
            prediction = model.predict(processed_data)[0]
            
            # Display prediction
            st.markdown("---")
            st.header("Prediction Result")
            
            if prediction == 1:
                st.markdown("""
                    <div class="prediction-box positive">
                        <h2 style="color: #d32f2f;">HIGH RISK - Heart Disease Detected</h2>
                        <p>The model predicts a high likelihood of heart disease based on the provided parameters.</p>
                        <p><strong>Recommendation:</strong> Immediate consultation with a cardiologist is strongly advised.</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class="prediction-box negative">
                        <h2 style="color: #388e3c;">LOW RISK - No Heart Disease Detected</h2>
                        <p>The model predicts a low likelihood of heart disease based on the provided parameters.</p>
                        <p><strong>Recommendation:</strong> Continue regular health check-ups and maintain a healthy lifestyle.</p>
                    </div>
                """, unsafe_allow_html=True)
            
            # Display probability if available
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(processed_data)[0]
                st.subheader("Confidence Levels")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("No Heart Disease", f"{proba[0]*100:.2f}%")
                with col2:
                    st.metric("Heart Disease", f"{proba[1]*100:.2f}%")
                    
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
            st.info("Please ensure all input values are valid and the model file is properly trained.")

def show_about_page():
    st.header("About the Model")
    
    st.markdown("""
    <div class="info-box">
    <h3>Model Overview</h3>
    <p>This heart disease prediction system uses machine learning algorithms to assess the risk of heart disease 
    based on various clinical parameters. The system was trained on the UCI Heart Disease dataset.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("Algorithms Used")
    st.markdown("""
    The following machine learning algorithms were evaluated:
    - K-Nearest Neighbors (KNN)
    - Support Vector Classifier (SVC)
    - Decision Tree Classifier
    - Random Forest Classifier
    
    The best performing model was selected based on accuracy, precision, recall, and F1-score metrics.
    """)
    
    st.subheader("Model Performance")
    st.markdown("""
    The selected model achieved:
    - High accuracy on test data
    - Balanced precision and recall
    - Robust performance across different patient demographics
    
    Note: This model was trained using GridSearchCV for hyperparameter optimization.
    """)

def show_feature_info_page():
    st.header("Feature Information")
    
    features_info = {
        "Age": "Age of the patient in years",
        "Sex": "Gender of the patient (0 = Female, 1 = Male)",
        "Chest Pain Type (cp)": """
            - Type 0: Typical Angina
            - Type 1: Atypical Angina
            - Type 2: Non-anginal Pain
            - Type 3: Asymptomatic
        """,
        "Resting Blood Pressure (trestbps)": "Resting blood pressure in mm Hg on admission to the hospital",
        "Serum Cholesterol (chol)": "Serum cholesterol level in mg/dl",
        "Fasting Blood Sugar (fbs)": "Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)",
        "Resting ECG (restecg)": """
            - 0: Normal
            - 1: ST-T wave abnormality
            - 2: Left ventricular hypertrophy
        """,
        "Maximum Heart Rate (thalach)": "Maximum heart rate achieved during exercise",
        "Exercise Induced Angina (exang)": "Exercise induced angina (1 = yes; 0 = no)",
        "ST Depression (oldpeak)": "ST depression induced by exercise relative to rest",
        "Slope": """
            - 0: Upsloping
            - 1: Flat
            - 2: Downsloping
        """,
        "Number of Major Vessels (ca)": "Number of major vessels colored by fluoroscopy (0-4)",
        "Thalassemia (thal)": """
            - 0: Normal
            - 1: Fixed defect
            - 2: Reversible defect
            - 3: Unknown
        """
    }
    
    for feature, description in features_info.items():
        with st.expander(feature):
            st.markdown(description)

if __name__ == "__main__":
    main()