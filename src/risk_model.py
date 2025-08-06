import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.utils import resample
import joblib

df = pd.read_csv("data/maternal_health_risk.csv")

df["Headache"] =  (df["SystolicBP"] > 120).astype(int)*5
df["BlurredVision"] = (df["DiastolicBP"] > 90).astype(int)*7
df["Convulsions"] = (df["BS"] > 100).astype(int)*10
df["Swelling"] = (df["BodyTemp"] > 37).astype(int)*6


def brain_risk(row):
    score =  row["Headache"] + row["BlurredVision"] + row["Convulsions"] + row["Swelling"]
    if score >= 13:
        return "High"
    elif score >=7:
        return "Medium"
    else:
        return "Low"
    
df["BrainRisk"] = df.apply(brain_risk, axis = 1)

print("Pre Balance: ", df["BrainRisk"].value_counts())

df_high = df[df.BrainRisk == "High"]
df_med = df[df.BrainRisk == "Medium"]
df_low = df[df.BrainRisk == "Low"]

balanced_dfs = [df_low]

if len(df_high) > 0:
    df_high_up = resample(df_high, replace=True, n_samples=len(df_low), random_state=42)
    balanced_dfs.append(df_high_up)

if len(df_med) > 0:
    df_med_up = resample(df_med, replace=True, n_samples=len(df_low), random_state=42)
    balanced_dfs.append(df_med_up)

df_balanced = pd.concat(balanced_dfs)
df_balanced["Headache"] =  (df_balanced["SystolicBP"] > 140).astype(int)*5
df_balanced["BlurredVision"] = (df_balanced["DiastolicBP"] > 90).astype(int)*7
df_balanced["Convulsions"] = (df_balanced["BS"] > 120).astype(int)*10
df_balanced["Swelling"] = (df_balanced["BodyTemp"] > 38).astype(int)*6

df_balanced["BrainRisk"] = df_balanced.apply(brain_risk, axis = 1)


print("Post Balance: ", df_balanced["BrainRisk"].value_counts())
x = df_balanced[["Headache", "BlurredVision", "Convulsions", "Swelling"]]
y = df_balanced["BrainRisk"]


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))

joblib.dump(model, "models/brain_risk_model.pkl")
print(" Model Saved")