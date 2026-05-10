# AI Impact on Jobs & Salary Trends

An end-to-end Data Mining and Machine Learning project analyzing how Artificial Intelligence is reshaping the global workforce, salary trends, job stability, and future career growth.

The project follows the complete CRISP-DM methodology and combines Machine Learning, Business Intelligence, interactive visualization, clustering, anomaly detection, and predictive analytics into a single professional analytics platform.

---

# Live Demo

Dashboard:  
https://dm-ai-jobs-project.streamlit.app/

GitHub Repository:  
https://github.com/Mariam-Nassar/dm-ai-jobs-project

---

# Team Members

- Mariam Khalil
- Samira Gamal

---

# Project Overview

Artificial Intelligence is rapidly transforming labor markets worldwide.

Some jobs are becoming more valuable due to increasing technology adoption, while others face high automation risk and declining demand.

This project uses Data Mining and Machine Learning techniques to analyze workforce transformation and answer important business questions such as:

- Which jobs are most vulnerable to AI automation?
- Which skills improve career stability?
- How does AI risk affect salaries?
- Which workforce groups are future-proof?
- How can organizations prepare for workforce transformation?

The platform provides predictive analytics and interactive business intelligence dashboards for workforce analysis and strategic decision-making.

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

# CRISP-DM Workflow

1. Business Understanding
2. Data Understanding
3. Data Preparation
4. Modeling
5. Evaluation
6. Deployment

---

# Dataset

Dataset Name:  
Future of Jobs AI Dataset

Source:  
Kaggle

---

# Dataset Description

- Records: 50,000+
- Time Range: 2015 — 2025
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

- Data cleaning
- Leakage removal
- Feature engineering
- Encoding categorical variables
- Feature scaling
- Outlier analysis
- Feature transformation
- Dataset splitting

Additional engineered features include:

- risk_demand_interaction
- human_capital_index
- ai_vulnerability
- openings_growth_rate

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

# Key Results

- Random Forest Classification Accuracy: 85%
- Macro F1-score: 0.83
- Random Forest Regression R²: 0.87
- 3 workforce clusters discovered
- Isolation Forest detected approximately 5% anomalies

---

# Interactive Dashboard

The project includes a fully interactive Streamlit dashboard for workforce intelligence and business analytics.

Dashboard Features:

- Interactive global filters
- Workforce risk analysis
- Salary intelligence analytics
- Country-level comparisons
- Classification prediction system
- Regression prediction engine
- Clustering visualization
- Anomaly detection insights
- PCA visualization
- Business intelligence insights

Dashboard Sections:

- Classification Panel
- Regression Panel
- Clustering Analysis
- Anomaly Explorer
- PCA Visualization
- Business Insights Panel

Live Dashboard:  
https://dm-ai-jobs-project.streamlit.app/

---

# Dashboard Preview

Add dashboard screenshots here.

Example:

![Dashboard Screenshot](assets/dashboard.png)

---

# How to Run the Project

## 1. Clone the Repository

```bash
git clone https://github.com/Mariam-Nassar/dm-ai-jobs-project.git
cd dm-ai-jobs-project
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## 3. Run the Dashboard

```bash
streamlit run app_professional.py
```

## 4. Open in Browser

```text
http://localhost:8501
```

---

# Technologies Used

## Programming & Analytics

- Python
- Pandas
- NumPy

## Machine Learning

- Scikit-learn

## Visualization & Dashboard

- Plotly
- Streamlit
- Matplotlib
- Seaborn

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
│   ├── preprocessing.ipynb
│   └── modeling_evaluation.ipynb
│
├── model_artifacts/
│
└── assets/
```

---

# Key Platform Modules

- Exploratory Data Analysis
- Classification
- Regression
- Clustering
- Anomaly Detection
- PCA Visualization
- Live Prediction Engine
- Business Insights

---

# Future Improvements

- Hyperparameter optimization
- Deep learning integration
- Real-time labor market APIs
- Explainable AI analytics
- Cloud scalability
- Real-world workforce datasets

---

# Conclusion

This project demonstrates how Data Mining and Machine Learning can support workforce intelligence and strategic decision-making in the age of Artificial Intelligence.

The platform combines predictive analytics, business intelligence, and interactive visualization into a unified analytics system.
