# Customer Churn Prediction – AI Assignment

## Overview
This project is a complete end-to-end AI solution built as part of an **AI Developer Assignment**. The goal of the project is to predict whether a customer is likely to churn (leave the service) based on customer demographic, service usage, and billing information.

The solution covers the full AI lifecycle:
- Data preprocessing
- Model training and evaluation
- Model explanation
- API deployment using FastAPI
- Proper documentation and GitHub-ready structure

---

## Dataset
The project uses the **Telco Customer Churn Dataset** obtained from Kaggle.

**Dataset characteristics:**
- Mixed data types (categorical + numerical)
- Missing values
- Real-world business use case

The raw dataset was manually downloaded as a CSV file and placed inside the `data/` directory.

---

## Task 1: Data Preprocessing
Data preprocessing was performed programmatically using Python to ensure reproducibility.

### Steps performed:
- Loaded raw CSV data using Pandas
- Removed non-informative columns (e.g., customer ID)
- Converted incorrect data types (e.g., TotalCharges)
- Handled missing values:
  - Numerical features → median imputation
  - Categorical features → mode imputation
- Encoded categorical variables using Label Encoding
- Applied feature scaling using StandardScaler
- Saved the cleaned dataset as `cleaned_telco_churn.csv`

The cleaned dataset contains no missing values and all features are numeric, making it suitable for machine learning models.

---

## Task 2: Model Building
Two machine learning classification models were trained and evaluated:

- Logistic Regression
- Random Forest Classifier

### Evaluation Metrics:
- Accuracy
- Precision
- Recall
- F1-score

### Model Selection:
Although both models achieved similar accuracy, **Logistic Regression** showed:
- Higher recall for the churn class
- Better F1-score
- Greater interpretability

Since churn prediction prioritizes identifying customers who are likely to leave, Logistic Regression was selected as the final model.

---

## Task 3: Model Explanation & Improvement Ideas

### Model Choice Explanation:
Logistic Regression was chosen due to its strong performance on imbalanced data, simplicity, and ease of interpretation. It correctly identified more churn cases compared to Random Forest.

### Feature Impact:
Feature impact was analyzed using model coefficients:
- Features such as **MonthlyCharges** showed a positive impact on churn
- **Tenure** and **Contract type** negatively correlated with churn, indicating customer loyalty

### Possible Improvements:
- Handle class imbalance using SMOTE or class weights
- Perform hyperparameter tuning (GridSearchCV)
- Use advanced models such as XGBoost or LightGBM
- Add more behavioral features for improved predictions

---

## Task 4: API Deployment
A REST API was developed using **FastAPI** to deploy the trained Logistic Regression model.

### API Features:
- Loads the trained model and feature order
- Accepts customer data as JSON
- Returns churn prediction with confidence score

### Endpoint:
- **POST** `/predict`

### Example Request Input:
```json
{
  "SeniorCitizen": 0,
  "tenure": -0.75,
  "MonthlyCharges": 0.42,
  "TotalCharges": -0.61,
  "gender": 1,
  "Partner": 0,
  "Dependents": 0,
  "PhoneService": 1,
  "MultipleLines": 0,
  "InternetService": 1,
  "OnlineSecurity": 0,
  "OnlineBackup": 1,
  "DeviceProtection": 0,
  "TechSupport": 0,
  "StreamingTV": 1,
  "StreamingMovies": 0,
  "Contract": 0,
  "PaperlessBilling": 1,
  "PaymentMethod": 2
}

### Example Response:
```json
{
  "prediction": "Churn",
  "confidence": 0.58
}
```

To ensure consistency between training and inference, the exact feature order used during training was saved and enforced during API prediction.

The API can be tested locally using Swagger UI available at:
```
http://127.0.0.1:8000/docs
```

---

## Project Structure
```
project/
│── data/
│   ├── Telco-Customer-Churn.csv
│   └── cleaned_telco_churn.csv
│── model/
│   ├── logistic_regression.pkl
│   └── feature_names.pkl
│── src/
│   ├── preprocessing.py
│   ├── train_model.py
│   └── api.py
│── requirements.txt
│── README.md
```

---

## How to Run the Project

### 1. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run preprocessing
```bash
python src/preprocessing.py
```

### 4. Train model
```bash
python src/train_model.py
```

### 5. Start API
```bash
uvicorn src.api:app --reload
```

---

## Conclusion
This project demonstrates a complete AI workflow from raw data to a deployed prediction API. It follows best practices in data preprocessing, model evaluation, deployment, and documentation, reflecting real-world AI development standards.
