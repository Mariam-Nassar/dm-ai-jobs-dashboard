# Workforce AI Risk Intelligence Platform

An end-to-end Data Mining and Machine Learning platform for analyzing the impact of Artificial Intelligence on jobs, salaries, workforce stability, and future career trends.

The project combines data analytics, predictive modeling, clustering, anomaly detection, dimensionality reduction, and interactive business intelligence dashboards into a single professional platform.

---

# Team Members

- Mariam Khalil
- Samira Gamal

---

# Project Overview

Artificial Intelligence is transforming the global workforce landscape.

Some professions are rapidly growing due to increasing technology adoption, while others face high automation risk and declining demand.

This project analyzes workforce trends using Machine Learning and Business Intelligence techniques to answer critical questions such as:

- Which jobs are most vulnerable to AI automation?
- Which skills increase career stability?
- How does AI risk affect salaries?
- Which workforce groups are future-proof?
- How can organizations prepare for workforce transformation?

The platform provides predictive analytics and interactive visualizations for workforce intelligence and strategic decision-making.

---

# Main Objectives

- Analyze the relationship between AI risk and job survival
- Identify high-risk and low-risk professions
- Study salary trends across countries and experience levels
- Predict workforce survival categories using classification models
- Predict salary patterns using regression models
- Discover workforce segments using clustering
- Detect abnormal workforce patterns using anomaly detection
- Reduce dimensionality using PCA visualization
- Build a professional interactive analytics dashboard

---

# Dataset

Dataset Name:
Future of Jobs AI Dataset

Source:
Kaggle

---

# Dataset Description

- Records: 17,000+
- Time Range: 2015 — 2024
- Multiple Countries Included
- Multiple Job Categories
- Workforce and AI-related indicators

---

# Main Features

- job_title
- country
- experience_level
- education_level
- year
- salary
- ai_risk_score
- primary_skill
- skill_demand_score
- job_openings

---

# Target Variables

## Classification Target

job_survival_class

Classes:

- 0 → At Risk
- 1 → Stable
- 2 → Growing

## Regression Target

salary

Used for salary prediction models.

---

# Data Preparation

The preprocessing pipeline includes:

- Missing value handling
- Data cleaning
- Feature engineering
- Encoding categorical variables
- Feature scaling
- Outlier analysis
- Feature transformation
- Dataset splitting

Additional engineered features include:

- Risk-demand interaction
- Human capital index
- AI vulnerability score
- Openings growth rate

---

# Machine Learning Models

## Classification Models

### Logistic Regression

Used as a baseline classification model.

### Random Forest Classifier

Used for workforce survival prediction.

Evaluation Metrics:

- Accuracy
- Macro F1-score
- Confusion Matrix

---

## Regression Models

### Linear Regression

Baseline salary prediction model.

### Random Forest Regressor

Advanced nonlinear salary prediction model.

Evaluation Metrics:

- RMSE
- MAE
- R² Score

---

# Clustering

K-Means clustering was applied to identify workforce segments based on:

- Salary
- AI risk
- Skill demand
- Job openings

The clustering process identifies different workforce profiles such as:

- Vulnerable roles
- Transitional roles
- Future-proof roles

---

# Anomaly Detection

Isolation Forest was used to detect unusual workforce records and abnormal career patterns.

This helps identify:

- Uncommon salary structures
- High-risk outlier jobs
- Rare workforce conditions

---

# PCA Dimensionality Reduction

Principal Component Analysis (PCA) was applied to:

- Reduce dimensionality
- Visualize workforce patterns
- Improve interpretability
- Explore feature relationships

---

# Interactive Dashboard Features

The Streamlit dashboard includes:

- Professional dark-themed UI
- Interactive global filters
- KPI analytics cards
- Workforce risk analysis
- Salary intelligence analytics
- Country-level comparisons
- Clustering visualization
- Anomaly detection insights
- PCA visualization
- Real-time prediction engine
- Business intelligence insights

---

# Business Intelligence Capabilities

The platform generates insights related to:

- Workforce transformation
- AI-driven job disruption
- Skill demand forecasting
- Salary intelligence
- Career sustainability
- Workforce segmentation

---

# Technologies Used

## Programming & Analytics

- Python
- Pandas
- NumPy

## Machine Learning

- Scikit-learn

## Visualization

- Plotly
- Streamlit

## Development Tools

- Git
- GitHub
- Jupyter Notebook
- VS Code

---

# Project Structure

```text
dm-ai-jobs-project/
│
├── app_professional.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── prepared_dataset_full.csv
│   ├── prepared_dataset_full.xls
│   └── prepared_dataset_topk.xls
│
├── notebooks/
│
├── model_artifacts/
│
└── assets/
```

## How to Run the Project

1. Install Dependencies

   pip install -r requirements.txt
   
2. Run the Dashboard
   
   streamlit run app_professional.py


## Key Platform Modules

Exploratory Data Analysis

Classification

Regression

Clustering

Anomaly Detection

PCA Visualization

Live Prediction Engine

Business Insights


## Future Improvements

Hyperparameter optimization

Deep learning integration

Online deployment

Real-time labor market APIs

More countries and industries

Explainable AI analytics


## Conclusion

This platform demonstrates how Data Mining and Machine Learning can support workforce intelligence and strategic decision-making in the age of Artificial Intelligence.

The project combines predictive analytics, business intelligence, and interactive visualization into a unified professional analytics system.


