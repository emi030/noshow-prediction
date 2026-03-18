import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

# ── Load Data ──
df = pd.read_csv('noshow_data.csv', skiprows=1)
print(df.head())
print(df.shape)
print(df.columns.tolist())

# ── Clean Data ──
df['patient age'] = pd.to_numeric(df['patient age'], errors='coerce')

# Create no-show indicator
# No-show = not cancelled, not seen
df['is_noshow'] = ((df['cancelled slots'] == 0) &
                   (df['sum appntmnts seen'] == 0)).astype(int)

print(df['is_noshow'].value_counts())
print(f"No-show rate: {df['is_noshow'].mean():.2%}")

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

# ── Statistical Tests ──
print("\n=== Statistical Tests ===")

# Insurance type vs no-show
ct_insurance = pd.crosstab(df['appt ins pkg type'], df['is_noshow'])
chi2, p_insurance, dof, _ = stats.chi2_contingency(ct_insurance)
significance = "SIGNIFICANT" if p_insurance < 0.05 else "not significant"
print(f"Insurance Type vs No-Show: p = {p_insurance:.4e} → {significance}")

# Sex vs no-show
ct_sex = pd.crosstab(df['patientsex'], df['is_noshow'])
chi2_sex, p_sex, dof_sex, _ = stats.chi2_contingency(ct_sex)
significance = "SIGNIFICANT" if p_sex < 0.05 else "not significant"
print(f"Sex vs No-Show: p = {p_sex:.4f} → {significance}")

# Appointment type vs no-show
top_appt = df['appttype'].value_counts().head(10).index
df_top = df[df['appttype'].isin(top_appt)]
ct_appt = pd.crosstab(df_top['appttype'], df_top['is_noshow'])
chi2_appt, p_appt, dof_appt, _ = stats.chi2_contingency(ct_appt)
significance = "SIGNIFICANT" if p_appt < 0.05 else "not significant"
print(f"Appointment Type vs No-Show: p = {p_appt:.4e} → {significance}")

# Age vs no-show
age_show = df[df['is_noshow'] == 0]['patient age'].dropna()
age_noshow = df[df['is_noshow'] == 1]['patient age'].dropna()
u_stat, p_age = stats.mannwhitneyu(age_noshow, age_show, alternative='two-sided')
significance = "SIGNIFICANT" if p_age < 0.05 else "not significant"
print(f"Age vs No-Show: mean showed={age_show.mean():.1f}, no-show={age_noshow.mean():.1f}, p = {p_age:.4e} → {significance}")

# ── Confirmation Effect Analysis ──
def classify_confirmation(val):
    if pd.isna(val) or str(val).strip() in ['-None-', '']:
        return 'No Contact'
    val = str(val).lower()
    if 'confirmed' in val or 'machine' in val or 'contacted' in val:
        return 'Confirmed'
    return 'No Contact'

df['conf_status'] = df['latestappconfresult'].apply(classify_confirmation)

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

ct_conf = pd.crosstab(df['conf_status'], df['is_noshow'])
chi2_conf, p_conf, dof_conf, _ = stats.chi2_contingency(ct_conf)
significance = "SIGNIFICANT" if p_conf < 0.05 else "not significant"
print(f"Confirmation vs No-Show: p = {p_conf:.4e} → {significance}")

# ── Scheduling Channel Analysis ──
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

ct_channel = pd.crosstab(df['sched_channel'], df['is_noshow'])
chi2_ch, p_ch, dof_ch, _ = stats.chi2_contingency(ct_channel)
significance = "SIGNIFICANT" if p_ch < 0.05 else "not significant"
print(f"Scheduling Channel vs No-Show: p = {p_ch:.4e} → {significance}")

# ── Machine Learning Model ──
ml = df[['patient age', 'patientsex', 'appt ins pkg type',
         'appttype', 'conf_status', 'sched_channel', 'is_noshow']].copy()

top8 = df['appttype'].value_counts().head(8).index
ml['appttype'] = ml['appttype'].apply(lambda x: x if x in top8 else 'Other')
ml = ml.dropna()

for col in ['patientsex', 'appt ins pkg type', 'appttype', 'conf_status', 'sched_channel']:
    ml[col] = LabelEncoder().fit_transform(ml[col].astype(str))

X = ml.drop('is_noshow', axis=1)
y = ml['is_noshow']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Logistic Regression Accuracy: {accuracy:.2%}")

# Random Forest
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
print(f"Random Forest Accuracy: {rf_accuracy:.2%}")

# ── Plot 5: Feature Importance ──
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(8, 5))
importances.plot(kind='bar', color='teal')
plt.title('Feature Importance — Random Forest')
plt.ylabel('Importance')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('noshow_feature_importance.png')
print("Feature importance plot saved!")