# Heart Disease Prediction System

A machine learning-based web application for predicting heart disease risk using clinical parameters.

## Features

- Interactive web interface built with Streamlit
- Multiple machine learning algorithms (KNN, SVC, Decision Tree, Random Forest)
- Hyperparameter-tuned models for optimal performance
- Real-time predictions with confidence scores
- Comprehensive feature information and model documentation

## Project Structure

```
heart-disease-prediction/
│
├── app.py                          # Main Streamlit application
├── HeartDiseasePredictionProject.ipynb  # Model training notebook
├── dataset.csv                     # Heart disease dataset
├── best_heart_model.pkl           # Trained model file
├── model_features.pkl             # Feature names for the model
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation

### Step 1: Clone or Download the Repository

```bash
git clone <repository-url>
cd heart-disease-prediction
```

Or download and extract the ZIP file to your desired location.

### Step 2: Create a Virtual Environment (Recommended)

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Train the Model (If not already trained)

If you don't have the `best_heart_model.pkl` file, run the Jupyter notebook:

```bash
jupyter notebook HeartDiseasePredictionProject.ipynb
```

Execute all cells to train the model and generate the pickle files.

## Requirements.txt

Create a `requirements.txt` file with the following content:

```
streamlit==1.31.0
pandas==2.1.4
numpy==1.26.3
scikit-learn==1.4.0
matplotlib==3.8.2
seaborn==0.13.1
joblib==1.3.2
```

## Running the Application

### Start the Streamlit App

```bash
streamlit run app.py
```

The application will automatically open in your default web browser at `http://localhost:8501`

### Alternative: Specify Port

```bash
streamlit run app.py --server.port 8080
```

## Troubleshooting

### Common Issues

**Issue**: ModuleNotFoundError
```
Solution: Ensure all dependencies are installed
pip install -r requirements.txt
```

**Issue**: Model file not found
```
Solution: Train the model using the Jupyter notebook
jupyter notebook HeartDiseasePredictionProject.ipynb
```

**Issue**: Streamlit command not found
```
Solution: Ensure virtual environment is activated
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

**Issue**: Port already in use
```
Solution: Use a different port
streamlit run app.py --server.port 8080
```
