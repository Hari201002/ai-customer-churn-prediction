import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "Telco-Customer-Churn.csv")

df = pd.read_csv(DATA_PATH)

print("Initial shape:", df.shape)


print("\nData Info:")
print(df.info())

print("\nMissing values per column:")
print(df.isnull().sum())



if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")


if "customerID" in df.columns:
    df.drop("customerID", axis=1, inplace=True)



numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns

for col in numerical_cols:
    df[col].fillna(df[col].median(), inplace=True)


categorical_cols = df.select_dtypes(include=["object"]).columns

for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

print("\nMissing values after handling:")
print(df.isnull().sum())


label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le


scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])


print("\nFinal shape after preprocessing:", df.shape)
print("\nSample data:")
print(df.head())


OUTPUT_PATH = os.path.join(BASE_DIR, "..", "data", "cleaned_telco_churn.csv")
df.to_csv(OUTPUT_PATH, index=False)

print("\nCleaned dataset saved at:", OUTPUT_PATH)





