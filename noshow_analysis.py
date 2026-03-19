import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from scipy import stats
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier


# ── Data ──────────────────────────────────────────────────────────────────────

df = pd.read_csv('noshow_data_v2.csv', skiprows=1, low_memory=False)

df['patient age'] = pd.to_numeric(df['patient age'], errors='coerce')
df['appt insexpctcopay'] = pd.to_numeric(df['appt insexpctcopay'], errors='coerce')

# EHR's native no-show flag had only 4 records — unusable
# No-show defined as: not cancelled and not seen
df['is_noshow'] = (
    (df['cancelled slots'] == 0) &
    (df['sum appntmnts seen'] == 0)
).astype(int)

print(f"Records: {len(df):,} | No-show rate: {df['is_noshow'].mean():.2%}")


# ── Feature Engineering ───────────────────────────────────────────────────────

def encode_confirmation(val):
    if pd.isna(val) or str(val).strip() in ['-None-', '']:
        return 'no_contact'
    val = str(val).lower()
    if any(x in val for x in ['confirmed', 'machine', 'contacted']):
        return 'confirmed'
    return 'no_contact'


def encode_channel(val):
    if pd.isna(val):
        return 'unknown'
    val = str(val).strip().upper()
    if val == 'PORTAL':
        return 'portal'
    if val.startswith('API-'):
        return 'api'
    return 'staff'


CHRONIC_ICD = ['I10', 'E11', 'E10', 'E78', 'E66', 'J44', 'I25', 'N18', 'F32', 'F41']
ICD_COLS = [f'icd10claimdiagcode0{i}' for i in range(1, 7)]

def flag_chronic(row):
    for col in ICD_COLS:
        val = str(row.get(col) or '')
        if any(val.startswith(c) for c in CHRONIC_ICD):
            return 1
    return 0


def bucket_copay(val):
    if pd.isna(val):
        return 'unknown'
    if val == 0:
        return '$0'
    if val <= 20:
        return '$1-20'
    if val <= 50:
        return '$21-50'
    return '$50+'


df['conf_status']   = df['latestappconfresult'].apply(encode_confirmation)
df['sched_channel'] = df['firstapptschdby'].apply(encode_channel)
df['has_chronic']   = df.apply(flag_chronic, axis=1)
df['copay_bucket']  = df['appt insexpctcopay'].apply(bucket_copay)
df['has_email']     = (df['validemail'].str.strip().str.upper() == 'Y').astype(int)

print(f"Chronic condition prevalence: {df['has_chronic'].mean():.2%}")


# ── Exploratory Analysis ──────────────────────────────────────────────────────

def save_bar(series, title, fname, figsize=(10, 5), color='steelblue'):
    fig, ax = plt.subplots(figsize=figsize)
    series.plot(kind='bar', ax=ax, color=color)
    ax.set_title(title)
    ax.set_ylabel('No-Show Rate')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()


save_bar(
    df.groupby('appt ins pkg type')['is_noshow'].mean().sort_values(ascending=False),
    'No-Show Rate by Insurance Type', 'noshow_by_insurance.png'
)
save_bar(
    df.groupby('appttype')['is_noshow'].mean().sort_values(ascending=False).head(10),
    'No-Show Rate by Appointment Type (Top 10)', 'noshow_by_appttype.png',
    color='coral'
)
save_bar(
    df.groupby('conf_status')['is_noshow'].mean() * 100,
    'No-Show Rate by Confirmation Status', 'noshow_by_confirmation.png',
    color=['#2ecc71', '#e74c3c']
)
save_bar(
    df.groupby('sched_channel')['is_noshow'].mean() * 100,
    'No-Show Rate by Scheduling Channel', 'noshow_by_channel.png',
    figsize=(8, 5)
)
save_bar(
    df.groupby('has_chronic')['is_noshow'].mean()
      .rename({0: 'None', 1: 'Chronic'}) * 100,
    'No-Show Rate by Chronic Condition', 'noshow_by_chronic.png',
    color=['#e74c3c', '#3498db']
)
save_bar(
    df.groupby('copay_bucket')['is_noshow'].mean()
      .reindex(['$0', '$1-20', '$21-50', '$50+', 'unknown'], fill_value=np.nan) * 100,
    'No-Show Rate by Copay', 'noshow_by_copay.png', color='mediumpurple'
)
save_bar(
    df.groupby('has_email')['is_noshow'].mean()
      .rename({0: 'No Email', 1: 'Has Email'}) * 100,
    'No-Show Rate by Valid Email', 'noshow_by_email.png',
    color=['#e67e22', '#27ae60']
)

fig, ax = plt.subplots(figsize=(10, 5))
for label, grp in df.groupby('is_noshow')['patient age']:
    grp.dropna().plot(kind='hist', bins=20, alpha=0.5,
                      label='No-Show' if label else 'Showed Up', ax=ax)
ax.set_xlabel('Patient Age')
ax.legend()
ax.set_title('Age Distribution by No-Show Status')
plt.tight_layout()
plt.savefig('noshow_by_age.png')
plt.close()

print("Plots saved.")


# ── Statistical Tests ─────────────────────────────────────────────────────────

def chi2_test(df, col):
    ct = pd.crosstab(df[col], df['is_noshow'])
    chi2, p, *_ = stats.chi2_contingency(ct)
    print(f"  {col:<35} chi2={chi2:.1f}  p={p:.2e}  "
          f"[{'p < 0.05' if p < 0.05 else 'ns'}]")


print("\nChi-square tests:")
for col in ['appt ins pkg type', 'patientsex', 'conf_status',
            'sched_channel', 'copay_bucket']:
    chi2_test(df, col)

top10 = df['appttype'].value_counts().head(10).index
chi2_test(df[df['appttype'].isin(top10)], 'appttype')
chi2_test(df.assign(has_chronic=df['has_chronic'].astype(str)), 'has_chronic')
chi2_test(df.assign(has_email=df['has_email'].astype(str)), 'has_email')

u, p = stats.mannwhitneyu(
    df.loc[df['is_noshow'] == 1, 'patient age'].dropna(),
    df.loc[df['is_noshow'] == 0, 'patient age'].dropna(),
    alternative='two-sided'
)
print(f"\n  Age Mann-Whitney  U={u:.0f}  p={p:.2e}")
print(f"  Mean age — showed: {df.loc[df['is_noshow']==0,'patient age'].mean():.1f} "
      f"| no-show: {df.loc[df['is_noshow']==1,'patient age'].mean():.1f}")


# ── Modelling ─────────────────────────────────────────────────────────────────

FEATURES = [
    'patient age', 'patientsex', 'appt ins pkg type', 'appttype',
    'conf_status', 'sched_channel', 'has_chronic', 'copay_bucket', 'has_email'
]

ml = df[FEATURES + ['is_noshow']].copy()

top8 = ml['appttype'].value_counts().head(8).index
ml['appttype'] = ml['appttype'].where(ml['appttype'].isin(top8), other='other')

n_before = len(ml)
ml = ml.dropna()
print(f"\nModelling set: {len(ml):,} rows | dropped {n_before - len(ml):,} missing")

CAT_COLS = ['patientsex', 'appt ins pkg type', 'appttype',
            'conf_status', 'sched_channel', 'copay_bucket']
for col in CAT_COLS:
    ml[col] = LabelEncoder().fit_transform(ml[col].astype(str))

X = ml.drop('is_noshow', axis=1)
y = ml['is_noshow']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# SMOTE on training split only
X_tr, y_tr = SMOTE(random_state=42).fit_resample(X_train, y_train)
print(f"After SMOTE: {pd.Series(y_tr).value_counts().to_dict()}")


def evaluate(name, model, X_tr, y_tr, X_te, y_te):
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    proba = model.predict_proba(X_te)[:, 1]
    print(f"\n{name}")
    print(f"  Accuracy : {accuracy_score(y_te, preds):.4f}")
    print(f"  ROC-AUC  : {roc_auc_score(y_te, proba):.4f}")
    print(classification_report(y_te, preds, digits=3))
    return model


evaluate("Logistic Regression",
         LogisticRegression(max_iter=1000, random_state=42),
         X_tr, y_tr, X_test, y_test)

evaluate("Random Forest",
         RandomForestClassifier(n_estimators=100, random_state=42),
         X_tr, y_tr, X_test, y_test)

evaluate("XGBoost",
         XGBClassifier(n_estimators=100, random_state=42,
                       eval_metric='logloss', verbosity=0),
         X_tr, y_tr, X_test, y_test)


# ── Hyperparameter Tuning ─────────────────────────────────────────────────────

param_grid = {
    'n_estimators':      [100, 200],
    'max_depth':         [5, 10, None],
    'min_samples_split': [2, 5],
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='roc_auc',
    n_jobs=-1,
    verbose=0,
)
grid.fit(X_tr, y_tr)

print(f"\nBest params : {grid.best_params_}")
print(f"CV AUC      : {grid.best_score_:.4f}")

best_rf = grid.best_estimator_
preds   = best_rf.predict(X_test)
proba   = best_rf.predict_proba(X_test)[:, 1]
print(f"Test AUC    : {roc_auc_score(y_test, proba):.4f}")
print(f"Test Acc    : {accuracy_score(y_test, preds):.4f}")
print(classification_report(y_test, preds, digits=3))


# ── Feature Importance ────────────────────────────────────────────────────────

importances = (
    pd.Series(best_rf.feature_importances_, index=X.columns)
    .sort_values(ascending=False)
)

fig, ax = plt.subplots(figsize=(10, 5))
importances.plot(kind='bar', ax=ax, color='teal')
ax.set_title('Feature Importance — Tuned Random Forest')
ax.set_ylabel('Importance')
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig('noshow_feature_importance.png')
plt.close()

print("\nTop features:")
print(importances.head(5).to_string())


# ── Save ────
joblib.dump(best_rf, 'noshow_model.pkl')
print("\nSaved → noshow_model.pkl")