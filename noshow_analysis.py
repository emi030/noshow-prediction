import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from scipy import stats
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# ── Load Data ──
df = pd.read_csv('noshow_data_v2.csv', skiprows=1, low_memory=False)
print(df.shape)
print(df.columns.tolist())

# ── Clean Data ──
df['patient age'] = pd.to_numeric(df['patient age'], errors='coerce')
df['appt insexpctcopay'] = pd.to_numeric(df['appt insexpctcopay'], errors='coerce')

# Create no-show indicator
# No-show = not cancelled, not seen
df['is_noshow'] = ((df['cancelled slots'] == 0) &
                   (df['sum appntmnts seen'] == 0)).astype(int)

print(df['is_noshow'].value_counts())
print(f"No-show rate: {df['is_noshow'].mean():.2%}")

# ── Feature Engineering ──

# Confirmation status
def classify_confirmation(val):
    if pd.isna(val) or str(val).strip() in ['-None-', '']:
        return 'No Contact'
    val = str(val).lower()
    if 'confirmed' in val or 'machine' in val or 'contacted' in val:
        return 'Confirmed'
    return 'No Contact'

df['conf_status'] = df['latestappconfresult'].apply(classify_confirmation)

# Scheduling channel
def classify_channel(val):
    if pd.isna(val):
        return 'Unknown'
    val = str(val).strip().upper()
    if val == 'PORTAL':
        return 'Patient Portal'
    elif val.startswith('API-'):
        return 'API / External'
    return 'Staff'

df['sched_channel'] = df['firstapptschdby'].apply(classify_channel)

# Chronic disease flag from ICD-10 codes
CHRONIC_CODES = ['I10', 'E11', 'E10', 'E78', 'E66', 'J44', 'I25', 'N18', 'F32', 'F41']

def has_chronic(row):
    for col in ['icd10claimdiagcode01', 'icd10claimdiagcode02',
                'icd10claimdiagcode03', 'icd10claimdiagcode04',
                'icd10claimdiagcode05', 'icd10claimdiagcode06']:
        val = str(row.get(col, '') or '')
        for code in CHRONIC_CODES:
            if val.startswith(code):
                return 1
    return 0

df['has_chronic'] = df.apply(has_chronic, axis=1)
print(f"Patients with chronic condition: {df['has_chronic'].mean():.2%}")

# Copay bucket
def copay_bucket(val):
    if pd.isna(val): return 'Unknown'
    if val == 0:     return '$0'
    if val <= 20:    return '$1-20'
    if val <= 50:    return '$21-50'
    return '$50+'

df['copay_bucket'] = df['appt insexpctcopay'].apply(copay_bucket)

# Valid email
df['has_email'] = df['validemail'].apply(lambda x: 1 if str(x).strip().upper() == 'Y' else 0)

# ── Plot 1: No-show by Insurance Type ──
noshow_by_insurance = df.groupby('appt ins pkg type')['is_noshow'].mean().sort_values(ascending=False)

plt.figure(figsize=(12, 6))
noshow_by_insurance.plot(kind='bar', color='steelblue')
plt.title('No-Show Rate by Insurance Type')
plt.xlabel('Insurance Type')
plt.ylabel('No-Show Rate')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('noshow_by_insurance.png')
print("Insurance plot saved!")

# ── Plot 2: No-show by Appointment Type ──
noshow_by_appttype = df.groupby('appttype')['is_noshow'].mean().sort_values(ascending=False).head(10)

plt.figure(figsize=(12, 6))
noshow_by_appttype.plot(kind='bar', color='coral')
plt.title('No-Show Rate by Appointment Type (Top 10)')
plt.xlabel('Appointment Type')
plt.ylabel('No-Show Rate')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('noshow_by_appttype.png')
print("Appointment type plot saved!")

# ── Plot 3: No-show by Age ──
plt.figure(figsize=(10, 5))
df.groupby('is_noshow')['patient age'].hist(alpha=0.5, bins=20)
plt.xlabel('Patient Age')
plt.ylabel('Count')
plt.title('Age Distribution by No-Show Status')
plt.legend(['Showed Up', 'No-Show'])
plt.tight_layout()
plt.savefig('noshow_by_age.png')
print("Age plot saved!")

# ── Plot 4: No-show by Sex ──
noshow_by_sex = df.groupby('patientsex')['is_noshow'].mean()

plt.figure(figsize=(6, 5))
noshow_by_sex.plot(kind='bar', color=['lightblue', 'pink'])
plt.title('No-Show Rate by Sex')
plt.xlabel('Sex')
plt.ylabel('No-Show Rate')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('noshow_by_sex.png')
print("Sex plot saved!")

# ── Plot 5: Confirmation Effect ──
noshow_by_conf = df.groupby('conf_status')['is_noshow'].mean() * 100

plt.figure(figsize=(7, 5))
noshow_by_conf.plot(kind='bar', color=['#2ecc71', '#e74c3c'])
plt.title('No-Show Rate: Confirmed vs No Contact')
plt.xlabel('Confirmation Status')
plt.ylabel('No-Show Rate (%)')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('noshow_by_confirmation.png')
print("Confirmation plot saved!")

# ── Plot 6: Scheduling Channel ──
noshow_by_channel = df.groupby('sched_channel')['is_noshow'].mean() * 100

plt.figure(figsize=(8, 5))
noshow_by_channel.plot(kind='bar', color=['#3498db', '#9b59b6', '#e67e22'])
plt.title('No-Show Rate by Scheduling Channel')
plt.xlabel('Scheduling Channel')
plt.ylabel('No-Show Rate (%)')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('noshow_by_channel.png')
print("Channel plot saved!")

# ── Plot 7: Chronic Disease vs No-Show ──
noshow_by_chronic = df.groupby('has_chronic')['is_noshow'].mean() * 100
noshow_by_chronic.index = ['No Chronic Condition', 'Has Chronic Condition']

plt.figure(figsize=(7, 5))
noshow_by_chronic.plot(kind='bar', color=['#e74c3c', '#3498db'])
plt.title('No-Show Rate: Chronic Condition vs None')
plt.xlabel('')
plt.ylabel('No-Show Rate (%)')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('noshow_by_chronic.png')
print("Chronic disease plot saved!")

# ── Plot 8: Copay vs No-Show ──
copay_order = ['$0', '$1-20', '$21-50', '$50+', 'Unknown']
noshow_by_copay = df.groupby('copay_bucket')['is_noshow'].mean() * 100
noshow_by_copay = noshow_by_copay.reindex([c for c in copay_order if c in noshow_by_copay.index])

plt.figure(figsize=(8, 5))
noshow_by_copay.plot(kind='bar', color='mediumpurple')
plt.title('No-Show Rate by Expected Copay')
plt.xlabel('Copay Amount')
plt.ylabel('No-Show Rate (%)')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('noshow_by_copay.png')
print("Copay plot saved!")

# ── Plot 9: Valid Email vs No-Show ──
noshow_by_email = df.groupby('has_email')['is_noshow'].mean() * 100
noshow_by_email.index = ['No Valid Email', 'Has Valid Email']

plt.figure(figsize=(7, 5))
noshow_by_email.plot(kind='bar', color=['#e67e22', '#27ae60'])
plt.title('No-Show Rate by Valid Email Status')
plt.xlabel('')
plt.ylabel('No-Show Rate (%)')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('noshow_by_email.png')
print("Email plot saved!")

# ── Statistical Tests ──
print("\n=== Statistical Tests ===")

ct_insurance = pd.crosstab(df['appt ins pkg type'], df['is_noshow'])
chi2, p_insurance, dof, _ = stats.chi2_contingency(ct_insurance)
print(f"Insurance Type vs No-Show: p = {p_insurance:.4e} → {'SIGNIFICANT' if p_insurance < 0.05 else 'not significant'}")

ct_sex = pd.crosstab(df['patientsex'], df['is_noshow'])
chi2_sex, p_sex, dof_sex, _ = stats.chi2_contingency(ct_sex)
print(f"Sex vs No-Show: p = {p_sex:.4f} → {'SIGNIFICANT' if p_sex < 0.05 else 'not significant'}")

top_appt = df['appttype'].value_counts().head(10).index
df_top = df[df['appttype'].isin(top_appt)]
ct_appt = pd.crosstab(df_top['appttype'], df_top['is_noshow'])
chi2_appt, p_appt, dof_appt, _ = stats.chi2_contingency(ct_appt)
print(f"Appointment Type vs No-Show: p = {p_appt:.4e} → {'SIGNIFICANT' if p_appt < 0.05 else 'not significant'}")

age_show   = df[df['is_noshow'] == 0]['patient age'].dropna()
age_noshow = df[df['is_noshow'] == 1]['patient age'].dropna()
u_stat, p_age = stats.mannwhitneyu(age_noshow, age_show, alternative='two-sided')
print(f"Age vs No-Show: mean showed={age_show.mean():.1f}, no-show={age_noshow.mean():.1f}, p = {p_age:.4e} → {'SIGNIFICANT' if p_age < 0.05 else 'not significant'}")

ct_conf = pd.crosstab(df['conf_status'], df['is_noshow'])
chi2_conf, p_conf, dof_conf, _ = stats.chi2_contingency(ct_conf)
print(f"Confirmation vs No-Show: p = {p_conf:.4e} → {'SIGNIFICANT' if p_conf < 0.05 else 'not significant'}")

ct_chronic = pd.crosstab(df['has_chronic'], df['is_noshow'])
chi2_chronic, p_chronic, dof_chronic, _ = stats.chi2_contingency(ct_chronic)
print(f"Chronic Disease vs No-Show: p = {p_chronic:.4e} → {'SIGNIFICANT' if p_chronic < 0.05 else 'not significant'}")

ct_copay = pd.crosstab(df['copay_bucket'], df['is_noshow'])
chi2_copay, p_copay, dof_copay, _ = stats.chi2_contingency(ct_copay)
print(f"Copay vs No-Show: p = {p_copay:.4e} → {'SIGNIFICANT' if p_copay < 0.05 else 'not significant'}")

ct_email = pd.crosstab(df['has_email'], df['is_noshow'])
chi2_email, p_email, dof_email, _ = stats.chi2_contingency(ct_email)
print(f"Valid Email vs No-Show: p = {p_email:.4e} → {'SIGNIFICANT' if p_email < 0.05 else 'not significant'}")

ct_channel = pd.crosstab(df['sched_channel'], df['is_noshow'])
chi2_ch, p_ch, dof_ch, _ = stats.chi2_contingency(ct_channel)
print(f"Scheduling Channel vs No-Show: p = {p_ch:.4e} → {'SIGNIFICANT' if p_ch < 0.05 else 'not significant'}")

# ── Machine Learning ──
ml = df[['patient age', 'patientsex', 'appt ins pkg type',
         'appttype', 'conf_status', 'sched_channel',
         'has_chronic', 'copay_bucket', 'has_email',
         'is_noshow']].copy()

top8 = df['appttype'].value_counts().head(8).index
ml['appttype'] = ml['appttype'].apply(lambda x: x if x in top8 else 'Other')
ml = ml.dropna()

for col in ['patientsex', 'appt ins pkg type', 'appttype',
            'conf_status', 'sched_channel', 'copay_bucket']:
    ml[col] = LabelEncoder().fit_transform(ml[col].astype(str))

X = ml.drop('is_noshow', axis=1)
y = ml['is_noshow']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── SMOTE ──
sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
print(f"\nAfter SMOTE: {pd.Series(y_train_sm).value_counts().to_dict()}")

# Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train_sm, y_train_sm)
print(f"\nLogistic Regression Accuracy: {accuracy_score(y_test, model.predict(X_test)):.2%}")
print(f"Logistic Regression ROC-AUC:  {roc_auc_score(y_test, model.predict_proba(X_test)[:,1]):.4f}")

# Random Forest + SMOTE
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_sm, y_train_sm)
print(f"\nRandom Forest Accuracy: {accuracy_score(y_test, rf.predict(X_test)):.2%}")
print(f"Random Forest ROC-AUC:  {roc_auc_score(y_test, rf.predict_proba(X_test)[:,1]):.4f}")

# XGBoost + SMOTE
xgb = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
xgb.fit(X_train_sm, y_train_sm)
print(f"\nXGBoost Accuracy: {accuracy_score(y_test, xgb.predict(X_test)):.2%}")
print(f"XGBoost ROC-AUC:  {roc_auc_score(y_test, xgb.predict_proba(X_test)[:,1]):.4f}")

# ── GridSearch on Random Forest ──
print("\nRunning GridSearch (this takes 2-3 min)...")
params = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5]
}
grid = GridSearchCV(RandomForestClassifier(random_state=42),
                    params, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)
grid.fit(X_train_sm, y_train_sm)
print(f"Best params: {grid.best_params_}")
print(f"Best AUC (CV): {grid.best_score_:.4f}")

best_rf = grid.best_estimator_
print(f"\nBest RF Accuracy: {accuracy_score(y_test, best_rf.predict(X_test)):.2%}")
print(f"Best RF ROC-AUC:  {roc_auc_score(y_test, best_rf.predict_proba(X_test)[:,1]):.4f}")

# ── Plot 10: Feature Importance ──
importances = pd.Series(best_rf.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(10, 5))
importances.plot(kind='bar', color='teal')
plt.title('Feature Importance — Random Forest (Tuned)')
plt.ylabel('Importance')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('noshow_feature_importance.png')
print("Feature importance plot saved!")

# ── Save Best Model ──
joblib.dump(best_rf, 'noshow_model.pkl')
print("Model saved!")
