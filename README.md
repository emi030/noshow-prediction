# Patient No-Show Prediction Analysis

Analysis of patient appointment no-shows using real clinic data to identify risk factors and build a predictive model. Includes a daily risk tool for front desk staff.

## Dataset
- 52,973 appointment records (2023–2026)
- Source: De-identified EHR export from JinCare Wellness Center
- Raw data not included due to data use agreement
- No-show rate: 18.16%

## Key Findings
- Patients with no confirmation contact have a **29.63% no-show rate** vs **2.91%** for confirmed patients — a 10x difference
- Uninsured (cash pay) patients have a **77% no-show rate** vs 4–5% for Medicare patients
- Patients with **no valid email on file have a 77% no-show rate** vs 8% for patients with a valid email
- Patients **with chronic conditions (diabetes, hypertension, obesity) have near-zero no-show rates**, while patients without any chronic diagnosis have a 23% rate
- Younger patients (mean age 45.4) no-show more than older patients (mean age 49.1, p < 0.0001)
- Sex was the only non-significant predictor (p = 0.35)
- Staff-scheduled appointments have the lowest no-show rate (6.8%) vs Patient Portal (15.3%)

## Visualizations
- No-show rate by insurance type
- No-show rate by appointment type
- Age distribution by no-show status
- Confirmation reminder effect
- Scheduling channel analysis
- Chronic disease vs no-show
- Valid email vs no-show
- Copay vs no-show
- Feature importance

## Machine Learning Model

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| Logistic Regression | 59.14% | 0.7669 |
| Random Forest + SMOTE | 85.75% | 0.8263 |
| XGBoost + SMOTE | 84.35% | 0.8646 |
| **Random Forest (GridSearch tuned)** | **86.06%** | **0.8382** |

SMOTE was used to address class imbalance. GridSearch optimized hyperparameters with 5-fold cross-validation (CV AUC: 0.9617). Top predictors: patient age, appointment type, chronic disease status.

## Clinical Recommendations
- Prioritize phone/SMS outreach for uninsured patients and unconfirmed appointments
- Flag patients with no valid email on file for direct phone outreach
- Hospital discharge follow-ups and counseling appointments need extra reminders
- Use the risk tool to generate a daily high-risk list for front desk outreach

## Files
| File | Description |
|------|-------------|
| `noshow_analysis.py` | Full analysis: data cleaning, visualizations, statistical tests, ML model |
| `predict_noshow.py` | Command-line tool: paste daily schedule, outputs risk list + Excel file |
| `noshow_risk_tool.html` | Browser-based tool: paste schedule, generates risk list instantly — no coding required |

## How to Use the Daily Risk Tool
1. Open `noshow_risk_tool.html` in any browser (double-click the file)
2. Copy your daily schedule from athenaOne
3. Paste into the tool and click Generate
4. High-risk patients are flagged in red

## Tools
- Python, pandas, numpy, matplotlib, seaborn, scikit-learn, scipy, xgboost, imbalanced-learn
- HTML, CSS, JavaScript

## Author
Emi Rivera | Data Science Student | George Washington University
