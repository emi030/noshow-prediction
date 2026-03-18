# Patient No-Show Prediction Analysis

Analysis of patient appointment no-shows using real clinic data to identify risk factors and build a predictive model.

## Dataset
- 48,810 appointment records (2023-2026)
- Source: De-identified EHR export from JinCare Wellness Center
- Raw data not included due to data use agreement
- No-show rate: 19.11%

## Analysis

### Key Findings
- Patients with no confirmation contact have a 29.63% no-show rate vs 2.91% for confirmed patients — a 10x difference
- Uninsured patients (cash pay) have a 77% no-show rate vs 4-5% for Medicare patients
- Younger patients (mean age 45.4) no-show more than older patients (mean age 49.1, p < 0.0001)
- Sex was the only non-significant predictor (p = 0.35)
- Staff-scheduled appointments have the lowest no-show rate (6.8%) vs Patient Portal (15.3%)

### Visualizations
- No-show rate by insurance type
- No-show rate by appointment type
- Age distribution by no-show status
- Confirmation reminder effect
- Scheduling channel analysis
- Feature importance

### Machine Learning Model
- Logistic Regression Accuracy: 60.73%
- Random Forest Accuracy: 82.38%
- Top predictors: patient age, appointment type, confirmation status, insurance type

## Clinical Recommendations
- Prioritize phone/SMS outreach for uninsured patients and unconfirmed appointments
- Flag hospital discharge follow-ups and counseling appointments for extra reminders
- Use the risk model to generate a daily high-risk list for front desk outreach

## Tools
- Python
- pandas, numpy
- matplotlib, seaborn
- scikit-learn, scipy

## Author
Emi Rivera | Data Science Student | George Washington University
