import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

np.random.seed(42)

def create_dataset(n_samples=10000):
    age = np.random.randint(18, 45, n_samples)
    
    polyhydramnios = np.random.binomial(1, 0.04, n_samples)
    oligohydramnios = np.random.binomial(1, 0.03, n_samples)
    elevated_afp = np.random.binomial(1, 0.05, n_samples)
    diabetes = np.random.binomial(1, 0.08, n_samples)
    hypertension = np.random.binomial(1, 0.10, n_samples)
    seizures = np.random.binomial(1, 0.01, n_samples)
    infections = np.random.binomial(1, 0.03, n_samples)
    no_folic_acid = np.random.binomial(1, 0.25, n_samples)
    family_history = np.random.binomial(1, 0.02, n_samples)
    
    risk_score = (
        polyhydramnios * 3 +
        oligohydramnios * 2 +
        elevated_afp * 4 +
        diabetes * 2 +
        hypertension * 1 +
        seizures * 3 +
        infections * 5 +
        no_folic_acid * 4 +
        family_history * 6 +
        (age > 35).astype(int) * 2 +
        (age < 20).astype(int) * 1
    )
    
    brain_abnormality = (risk_score >= 8).astype(int)
    
    random_cases = np.random.binomial(1, 0.001, n_samples)
    brain_abnormality = np.maximum(brain_abnormality, random_cases)
    
    data = pd.DataFrame({
        'maternal_age': age,
        'polyhydramnios': polyhydramnios,
        'oligohydramnios': oligohydramnios,
        'elevated_afp': elevated_afp,
        'diabetes': diabetes,
        'hypertension': hypertension,
        'maternal_seizures': seizures,
        'infections': infections,
        'no_folic_acid': no_folic_acid,
        'family_history': family_history,
        'brain_abnormality': brain_abnormality
    })
    
    return data

print("Creating dataset...")
df = create_dataset(10000)

print(f"\nDataset shape: {df.shape}")
print(f"\nBrain abnormalities: {df['brain_abnormality'].sum()} ({df['brain_abnormality'].mean()*100:.2f}%)")
print(f"\nDataset preview:")
print(df.head())

print(f"\nFeature correlations with brain abnormality:")
correlations = df.corr()['brain_abnormality'].sort_values(ascending=False)
print(correlations[:-1])

features = ['maternal_age', 'polyhydramnios', 'oligohydramnios', 'elevated_afp', 
           'diabetes', 'hypertension', 'maternal_seizures', 'infections', 
           'no_folic_acid', 'family_history']

X = df[features]
y = df['brain_abnormality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining model...")
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"\nModel Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred))

feature_importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nFeature Importance:")
print(feature_importance)

joblib.dump(model, 'models/brain_risk_model.pkl')
print(f"\nModel saved!")

print(f"\nExample prediction:")
example = [[35, 1, 0, 1, 1, 0, 0, 0, 1, 0]]
prediction = model.predict(example)[0]
probability = model.predict_proba(example)[0][1]
print(f"Risk: {'High' if prediction else 'Low'}")
print(f"Probability: {probability:.4f}")
