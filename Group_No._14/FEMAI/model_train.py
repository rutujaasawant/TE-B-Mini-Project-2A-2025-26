import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from pandas.api.types import is_object_dtype
import os

# Create models directory
os.makedirs("models", exist_ok=True)

# Suppress pandas FutureWarning
pd.set_option('future.no_silent_downcasting', True)


def convert_age_to_numeric(age):
    """Convert categorical age ranges to numeric values."""
    if not isinstance(age, str):
        return age
    age = age.lower().strip()
    age_map = {
        'less than 20': 18,
        '20-25': 22.5,
        '25-30': 27.5,
        '30-35': 32.5,
        '35-44': 39.5,
        '45 and above': 45
    }
    return age_map.get(age, np.nan)


def preprocess_data(df, features, categorical_cols, bmi_cols=None):
    """Preprocess DataFrame: calculate BMI, encode categoricals, ensure numeric types."""
    df = df.copy()

    # Strip column names to remove leading/trailing spaces
    df.columns = df.columns.str.strip()

    # Calculate BMI if needed
    if bmi_cols and 'BMI' not in df.columns and all(col in df.columns for col in bmi_cols):
        height_col = bmi_cols[1]
        if 'cm' in height_col.lower():
            df['BMI'] = df[bmi_cols[0]] / ((df[height_col] * 0.01) ** 2)
        else:
            df['BMI'] = df[bmi_cols[0]] / ((df[height_col] * 0.3048) ** 2)
        df['BMI'] = pd.to_numeric(df['BMI'], errors='coerce').fillna(
            df['BMI'].median() if df['BMI'].notna().any() else 25)

    # Encode categorical columns
    for col in [c for c in categorical_cols if c in df.columns]:
        if is_object_dtype(df[col]):
            df[col] = df[col].str.strip().str.lower().replace({
                'yes': 1, 'no': 0, 'y': 1, 'n': 0,
                'always': 1, 'sometimes': 1, 'sometime': 1, 'never': 0,
                'r': 0, 'i': 1, '2': 0, '4': 1,
                'yes, not diagnosed by a doctor': 1, 'yes, diagnosed by a doctor': 1,
                'low': 0, 'moderate': 1, 'high': 2  # Added for Stress Level
            }).infer_objects(copy=False).fillna(0).astype(int)

    # Ensure numeric columns
    for col in ['Age', 'BMI', 'Miscarriages', 'StressLevel']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(
                df[col].median() if df[col].notna().any() else (25 if col in ['Age', 'BMI'] else 0))

    # Drop rows with missing features or target
    available_features = [f for f in features if f in df.columns]
    df = df.dropna(subset=available_features + (['Severity'] if 'Severity' in df.columns else []))
    return df


# -------------------------------
# TRAIN PCOD MODEL
# -------------------------------
print("Training PCOD model...")
try:
    pcod_data = pd.read_csv("archive/result_data.csv")
    pcod_data.columns = pcod_data.columns.str.strip()
except FileNotFoundError:
    print("simplified_pcod_data.csv not found. Using fallback data.")
    pcod_data = pd.DataFrame({
        'Age': np.random.randint(18, 45, 100),
        'BMI': np.random.uniform(18, 35, 100),
        'IrregularPeriod': np.random.choice([0, 1], 100),
        'WeightGain': np.random.choice([0, 1], 100),
        'HairGrowth': np.random.choice([0, 1], 100),
        'Acne': np.random.choice([0, 1], 100),
        'HairLoss': np.random.choice([0, 1], 100),
        'FamilyHistory': np.random.choice([0, 1], 100),
        'Pain': np.random.choice([0, 1], 100),
        'StressLevel': np.random.choice([0, 1, 2], 100),
        'InsulinResistance': np.random.choice([0, 1], 100),
        'Severity': np.random.choice(['Normal', 'High'], 100)
    })

# Feature mappings
pcod_feature_mapping = {
    '20) Do you feel stressed due to your increased weight?': 'WeightGain',
    'Family history of PCOD': 'FamilyHistory',
    '7) Do you experience any pain and cramps in your lower abdomen, lower back and legs?': 'Pain'
}
pcod_data = pcod_data.rename(columns=pcod_feature_mapping)

# Derive HairGrowth
if 'Signs of PCOD' in pcod_data.columns:
    pcod_data['HairGrowth'] = pcod_data['Signs of PCOD'].apply(
        lambda x: 1 if isinstance(x, str) and 'hirsutism' in x.lower().strip() else 0
    )

# Define features and categorical columns
pcod_features = ["Age", "BMI", "IrregularPeriod", "WeightGain", "HairGrowth", "Acne", "HairLoss", "FamilyHistory",
                 "Pain", "StressLevel", "InsulinResistance"]
categorical_cols = ["IrregularPeriod", "WeightGain", "HairGrowth", "Acne", "HairLoss", "FamilyHistory", "Pain",
                    "StressLevel", "InsulinResistance"]

# Add missing features
for feature in pcod_features:
    if feature not in pcod_data.columns:
        pcod_data[feature] = np.random.randint(18, 45, len(pcod_data)) if feature == 'Age' else \
            np.random.uniform(18, 35, len(pcod_data)) if feature == 'BMI' else \
                np.random.choice([0, 1, 2], len(pcod_data)) if feature == 'StressLevel' else \
                    np.random.choice([0, 1], len(pcod_data))

# Derive Severity
if 'Severity' not in pcod_data.columns and 'Are you taking any medications for PCOD?' in pcod_data.columns:
    pcod_data['Severity'] = pcod_data['Are you taking any medications for PCOD?'].apply(
        lambda x: 'High' if isinstance(x, str) and x.lower().strip() == 'yes' else 'Normal'
    )

# Preprocess data
print("PCOD data columns:", pcod_data.columns.tolist())
pcod_data = preprocess_data(pcod_data, pcod_features, categorical_cols)

# Extract features and target
X_pcod = pcod_data[pcod_features]
y_pcod = pcod_data["Severity"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_pcod, y_pcod, test_size=0.2, random_state=42)

# Scale features
scaler_pcod = StandardScaler()
X_train = scaler_pcod.fit_transform(X_train)
X_test = scaler_pcod.transform(X_test)

# Train model
model_pcod = LogisticRegression(max_iter=500, class_weight='balanced')
model_pcod.fit(X_train, y_train)

# Evaluate model
print("PCOD model performance:")
print(classification_report(y_test, model_pcod.predict(X_test), zero_division=0))

# Save model and scaler
with open("models/pcod_model.pkl", "wb") as f:
    pickle.dump(model_pcod, f)
with open("models/pcod_scaler.pkl", "wb") as f:
    pickle.dump(scaler_pcod, f)

# -------------------------------
# TRAIN PCOS MODEL
# -------------------------------
print("Training PCOS model...")
try:
    pcos_data = pd.read_csv("archive/simplified_pcos_data.csv")
    pcos_data.columns = pcos_data.columns.str.strip()
    try:
        pcos_data1 = pd.read_csv("archive/simplified_cleaned_data.csv")
        pcos_data1.columns = pcos_data1.columns.str.strip()
        # Map relevant columns from simplified_cleaned_data.csv
        if 'Age' in pcos_data1.columns:
            pcos_data1['Age'] = pcos_data1['Age'].apply(convert_age_to_numeric)
            if 'Age' in pcos_data.columns:
                pcos_data['Age'] = pcos_data['Age'].fillna(
                    pcos_data1['Age'].reindex(pcos_data.index, fill_value=pcos_data1['Age'].median()))
            else:
                pcos_data['Age'] = pcos_data1['Age'].reindex(pcos_data.index, fill_value=pcos_data1['Age'].median())
        if 'Family_History_PCOS' in pcos_data1.columns:
            pcos_data['FamilyHistory'] = pcos_data1['Family_History_PCOS'].reindex(pcos_data.index, fill_value=0)
        if 'Conception_Difficulty' in pcos_data1.columns:
            pcos_data['Infertility'] = pcos_data1['Conception_Difficulty'].reindex(pcos_data.index, fill_value=0)
        if 'Menstrual_Irregularity' in pcos_data1.columns:
            pcos_data['IrregularPeriods'] = pcos_data1['Menstrual_Irregularity'].reindex(pcos_data.index, fill_value=0)
        if 'Hirsutism' in pcos_data1.columns:
            pcos_data['HairGrowth'] = pcos_data1['Hirsutism'].reindex(pcos_data.index, fill_value=0)
        if 'Pimples' in pcos_data1.columns:
            pcos_data['Acne'] = pcos_data1['Pimples'].reindex(pcos_data.index, fill_value=0)
        if 'Hair_Loss' in pcos_data1.columns:
            pcos_data['HairLoss'] = pcos_data1['Hair_Loss'].reindex(pcos_data.index, fill_value=0)
        if 'Insulin_Resistance' in pcos_data1.columns:
            pcos_data['InsulinResistance'] = pcos_data1['Insulin_Resistance'].reindex(pcos_data.index, fill_value=0)
        if 'Stress_Level' in pcos_data1.columns:
            pcos_data['StressLevel'] = pcos_data1['Stress_Level'].reindex(pcos_data.index, fill_value=0)
    except FileNotFoundError:
        print("simplified_cleaned_data.csv not found. Using simplified_pcos_data.csv only.")
except FileNotFoundError:
    print("simplified_pcos_data.csv not found. Using fallback data.")
    pcos_data = pd.DataFrame({
        'Age': np.random.randint(18, 45, 100),
        'BMI': np.random.uniform(18, 35, 100),
        'IrregularPeriods': np.random.choice([0, 1], 100),
        'Infertility': np.random.choice([0, 1], 100),
        'Miscarriages': np.random.randint(0, 3, 100),
        'HairGrowth': np.random.choice([0, 1], 100),
        'Acne': np.random.choice([0, 1], 100),
        'HairLoss': np.random.choice([0, 1], 100),
        'FamilyHistory': np.random.choice([0, 1], 100),
        'StressLevel': np.random.choice([0, 1, 2], 100),
        'InsulinResistance': np.random.choice([0, 1], 100),
        'Severity': np.random.choice(['Normal', 'High'], 100)
    })

# Feature mappings
pcos_feature_mapping = {
    'Cycle(R/I)': 'IrregularPeriods',
    'Pregnant(Y/N)': 'Infertility',
    'No. of aborptions': 'Miscarriages',
    'hair growth(Y/N)': 'HairGrowth',
    'Pimples(Y/N)': 'Acne',
    'Hair loss(Y/N)': 'HairLoss',
    'Family history of PCOD': 'FamilyHistory',
    'Weight (Kg)': 'Weight_kg',
    'Height(Cm)': 'Height_cm'
}
pcos_data = pcos_data.rename(columns=pcos_feature_mapping)

# Define features and categorical columns
pcos_features = ["Age", "BMI", "IrregularPeriods", "Infertility", "Miscarriages", "HairGrowth", "Acne", "HairLoss",
                 "FamilyHistory", "StressLevel", "InsulinResistance"]
categorical_cols = ["IrregularPeriods", "Infertility", "HairGrowth", "Acne", "HairLoss", "FamilyHistory", "StressLevel",
                    "InsulinResistance"]

# Preprocess data
print("PCOS data columns:", pcos_data.columns.tolist())
pcos_data = preprocess_data(pcos_data, pcos_features, categorical_cols, bmi_cols=['Weight_kg', 'Height_cm'])

# Invert Infertility (if not already inverted)
if 'Infertility' in pcos_data.columns:
    pcos_data['Infertility'] = pd.to_numeric(pcos_data['Infertility'], errors='coerce').fillna(0).astype(int)
    pcos_data['Infertility'] = 1 - pcos_data['Infertility']

# Derive Severity
if 'Severity' not in pcos_data.columns and 'PCOS (Y/N)' in pcos_data.columns:
    pcos_data['Severity'] = pcos_data['PCOS (Y/N)'].apply(lambda x: 'High' if x == 1 else 'Normal')

# Ensure all features exist
for feature in pcos_features:
    if feature not in pcos_data.columns:
        pcos_data[feature] = np.random.randint(18, 45, len(pcos_data)) if feature == 'Age' else \
            np.random.uniform(18, 35, len(pcos_data)) if feature == 'BMI' else \
                np.random.randint(0, 3, len(pcos_data)) if feature == 'Miscarriages' else \
                    np.random.choice([0, 1, 2], len(pcos_data)) if feature == 'StressLevel' else \
                        np.random.choice([0, 1], len(pcos_data))

# Extract features and target
X_pcos = pcos_data[pcos_features]
y_pcos = pcos_data["Severity"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_pcos, y_pcos, test_size=0.2, random_state=42)

# Scale features
scaler_pcos = StandardScaler()
X_train = scaler_pcos.fit_transform(X_train)
X_test = scaler_pcos.transform(X_test)

# Train model
model_pcos = LogisticRegression(max_iter=500, class_weight='balanced')
model_pcos.fit(X_train, y_train)

# Evaluate model
print("PCOS model performance:")
print(classification_report(y_test, model_pcos.predict(X_test), zero_division=0))

# Save model and scaler
with open("models/pcos_model.pkl", "wb") as f:
    pickle.dump(model_pcos, f)
with open("models/pcos_scaler.pkl", "wb") as f:
    pickle.dump(scaler_pcos, f)

print("All models and scalers saved in 'models/' directory.")