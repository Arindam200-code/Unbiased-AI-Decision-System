# 🧠 Unbiased AI Decision System

An AI-powered system that detects and analyzes bias in decision-making across multiple domains like hiring, education, healthcare, and banking.

---

## 🚀 Overview

The Unbiased AI Decision System is designed to ensure fairness in machine learning models by identifying bias in predictions and providing transparency through visual analysis and reports.

This project demonstrates how biased datasets can influence AI decisions and how fairness metrics can be used to evaluate them.

---

## 🎯 Features

📊 Upload and analyze datasets (CSV)
⚙️ Train ML model (Random Forest)
🔍 Detect bias using fairness metrics
📈 Visualize:
Gender distribution
Selection rate
Feature importance
📄 Generate downloadable PDF report
🌐 Multi-domain support:
Job Hiring
Education
Medical Care
Bank Loan

---

## 🏗️ Project Structure

```
├── hiring.py           # Dataset generation (biased data)
├── preprocessing.py    # Data preprocessing & encoding
├── model.py            # ML model training
├── fairness.py         # Bias detection logic
├── app.py              # Streamlit web app
├── hiring.csv          # Sample dataset
└── README.md
```

---

## ⚙️ Technologies Used

Python
Streamlit
Pandas
NumPy
Scikit-learn
Fairlearn
Matplotlib
ReportLab

---

## 📊 How It Works

Upload dataset (or use generated biased dataset)
Preprocess data (encoding categorical features)
Train model using Random Forest
Evaluate:
Accuracy
Bias score (Demographic Parity)
Visualize results
Generate fairness report (PDF)

---

## 📈 Fairness Metric Used

Demographic Parity Difference

Measures difference in selection rates across groups
Ideal value: 0 (fair)
Threshold:
< 0.1 → Fair
> 0.1 → Bias detected

---

## 
