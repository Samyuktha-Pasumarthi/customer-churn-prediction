# Customer Churn Prediction

An end-to-end machine learning project to predict customer churn using feature engineering, model comparison, and explainability techniques.

---

## Problem Statement

Customer churn is a critical business problem where companies aim to identify customers likely to leave. This project builds a predictive model to classify customers as churned or not churned.

---

## Project Pipeline

1. **Exploratory Data Analysis (EDA)**
   - Data understanding and visualization
   - Distribution analysis and churn patterns

2. **Feature Engineering**
   - Created ratio-based features:
     - `balance_per_product`
     - `salary_balance_ratio`
   - Binned features:
     - `age_group`
     - `tenure_bucket`
   - Derived flags:
     - `high_balance`

3. **Data Preprocessing**
   - Handling missing values
   - Scaling numerical features
   - One-hot encoding categorical variables
   - Implemented using `ColumnTransformer` and `Pipeline`

4. **Model Training & Evaluation**
   - Models used:
     - Logistic Regression
     - Random Forest
     - Gradient Boosting
     - AdaBoost
     - SVM
     - XGBoost
   - Evaluated using:
     - ROC-AUC (primary metric)
     - Accuracy, Precision, Recall, F1-score
   - Used Stratified K-Fold Cross Validation

5. **Model Selection**
   - Best model: **Gradient Boosting**
   - Achieved highest ROC-AUC with stable performance

6. **Model Explainability**
   - Used SHAP (SHapley Additive Explanations)
   - Identified key drivers of churn:
     - Product usage
     - Age
     - Balance
     - Customer activity
     - Engineered features

7. **Inference**
   - Simulated predictions on new customer data
   - Generated churn probability for real-world use

8. **Model Deployment (Local)**
   - Saved trained pipeline using `joblib`
   - Reusable for predictions without retraining

---

## Key Insights

- Ensemble models outperform linear models, indicating non-linear relationships
- Customer engagement (`active_member`) strongly reduces churn
- High balance and multiple products increase churn risk
- Feature engineering significantly improved model performance

---

## Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- SHAP
- Matplotlib, Seaborn

---

## Project Structure
customer-churn-prediction/
│
├── data/
│ ├── raw/
│ │ └── Bank Customer Churn Prediction.csv
│ ├── processed/
│ │ └── processed_data.csv
│
├── notebooks/
│ ├── 01_eda.ipynb
│ ├── 02_feature_engineering.ipynb
│ ├── 03_model_training.ipynb
│ ├── 04_model_explainability.ipynb
│
├── models/
│ └── best_churn_pipeline.pkl
│
├── requirements.txt
├── README.md



---

## Example Prediction
Predicted churn: 0
Probability of churn: 0.029


This indicates a low likelihood of churn for the given customer profile.

---

## Conclusion

This project demonstrates a complete ML workflow from data exploration to model deployment, with strong emphasis on feature engineering and model interpretability.

---

## Future Improvements

- Hyperparameter tuning
- Deploy as web app (Streamlit/Flask)
- Real-time prediction API
- Handling class imbalance

---


