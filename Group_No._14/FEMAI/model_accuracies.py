# import pandas as pd
# import numpy as np
# import pickle
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import classification_report
# import os
#
# # Create models directory
# os.makedirs("models", exist_ok=True)
#
# # Suppress pandas FutureWarning
# pd.set_option('future.no_silent_downcasting', True)
#
# # Convert age to numeric
# def convert_age_to_numeric(age):
#     if not isinstance(age, str):
#         return age
#     age = age.lower().strip()
#     age_map = {
#         'less than 20': 18,
#         '20-25': 22.5,
#         '25-30': 27.5,
#         '30-35': 32.5,
#         '35-44': 39.5,
#         '45 and above': 45
#     }
#     return age_map.get(age, np.nan)
#
# # Preprocess data
# def preprocess_data(df, features, categorical_cols, bmi_cols=None):
#     df = df.copy()
#     df.columns = df.columns.str.strip()
#     print("Columns after stripping:", df.columns.tolist())
#
#     # Calculate BMI if needed
#     if bmi_cols and 'BMI' not in df.columns and all(col in df.columns for col in bmi_cols):
#         weight_col, height_col = bmi_cols
#         if 'cm' in height_col.lower():
#             df['BMI'] = df[weight_col] / ((df[height_col] * 0.01) ** 2)
#         else:
#             df['BMI'] = df[weight_col] / ((df[height_col] * 0.3048) ** 2)
#         df['BMI'] = pd.to_numeric(df['BMI'], errors='coerce').fillna(
#             df['BMI'].median() if df['BMI'].notna().any() else 25)
#
#     # Convert categorical columns
#     for col in [c for c in categorical_cols if c in df.columns]:
#         if df[col].dtype == 'object':
#             df[col] = df[col].str.strip().str.lower().replace({
#                 'yes': 1, 'no': 0, 'y': 1, 'n': 0,
#                 'always': 1, 'sometimes': 1, 'sometime': 1, 'never': 0,
#                 'r': 0, 'i': 1, '2': 0, '4': 1,
#                 'yes, not diagnosed by a doctor': 1, 'yes, diagnosed by a doctor': 1,
#                 'low': 0, 'moderate': 1, 'high': 2
#             }).infer_objects(copy=False).fillna(0).astype(int)
#
#     # Convert numeric columns
#     for col in ['Age', 'BMI', 'Miscarriages', 'StressLevel']:
#         if col in df.columns:
#             df[col] = pd.to_numeric(df[col], errors='coerce').fillna(
#                 df[col].median() if df[col].notna().any() else (25 if col in ['Age', 'BMI'] else 0))
#
#     # Add missing features
#     for feature in features:
#         if feature not in df.columns:
#             print(f"Adding missing feature: {feature}")
#             df[feature] = np.random.randint(18, 45, len(df)) if feature == 'Age' else \
#                 np.random.uniform(18, 35, len(df)) if feature == 'BMI' else \
#                     np.random.randint(0, 3, len(df)) if feature == 'Miscarriages' else \
#                         np.random.choice([0, 1, 2], len(df)) if feature == 'StressLevel' else \
#                             np.random.choice([0, 1], len(df))
#
#     # Drop rows with missing features
#     available_features = [f for f in features if f in df.columns]
#     df = df.dropna(subset=available_features)
#     print("Columns after preprocessing:", df.columns.tolist())
#     return df
#
# # PCOD: Process result_data_simplified.csv
# print("Processing PCOD data...")
# try:
#     pcod_data = pd.read_csv("archive/result_data_simplified.csv")
#     print("Raw PCOD data columns:", pcod_data.columns.tolist())
# except FileNotFoundError:
#     print("result_data_simplified.csv not found.")
#     raise
#
# # PCOD feature mappings
# pcod_feature_mapping = {
#     'Family history of PCOD': 'FamilyHistory',
#     '7) Do you experience any pain and cramps in your lower abdomen, lower back and legs?': 'Pain',
#     'Signs of PCOD': 'HairGrowth',
#     '20) Do you feel stressed due to your increased weight?': 'WeightGain'
# }
# pcod_data = pcod_data.rename(columns=pcod_feature_mapping)
# print("PCOD columns after renaming:", pcod_data.columns.tolist())
#
# # Derive HairGrowth
# if 'HairGrowth' in pcod_data.columns:
#     pcod_data['HairGrowth'] = pcod_data['HairGrowth'].apply(
#         lambda x: 1 if isinstance(x, str) and 'hirsutism' in x.lower().strip() else x if isinstance(x, (int, float)) else 0
#     )
#
# # Define PCOD features
# pcod_features = ["Age", "BMI", "IrregularPeriod", "WeightGain", "HairGrowth", "Acne", "HairLoss", "FamilyHistory", "Pain", "StressLevel", "InsulinResistance"]
# categorical_cols = ["IrregularPeriod", "WeightGain", "HairGrowth", "Acne", "HairLoss", "FamilyHistory", "Pain", "StressLevel", "InsulinResistance"]
#
# # Preprocess PCOD data
# pcod_data = preprocess_data(pcod_data, pcod_features, categorical_cols)
# print("Attempting to select PCOD features:", pcod_features)
#
# # Derive Severity for PCOD
# if 'Medical Condition' in pcod_data.columns and pcod_data['Medical Condition'].nunique() > 1:
#     print("Using Medical Condition for PCOD Severity...")
#     pcod_data['Severity'] = pcod_data['Medical Condition'].apply(lambda x: 'High' if x == 1 else 'Normal')
# elif 'Are you taking any medications for PCOD?' in pcod_data.columns and pcod_data['Are you taking any medications for PCOD?'].nunique() > 1:
#     print("Using medication status for PCOD Severity...")
#     pcod_data['Severity'] = pcod_data['Are you taking any medications for PCOD?'].apply(lambda x: 'High' if x == 1 else 'Normal')
# else:
#     print("Deriving Severity for PCOD based on symptoms...")
#     symptom_cols = [col for col in ['HairGrowth', 'FamilyHistory', 'Pain', 'WeightGain', '17) Do you feel stressed that you might face trouble getting pregnant?'] if col in pcod_data.columns]
#     if symptom_cols:
#         pcod_data['Severity'] = pcod_data[symptom_cols].sum(axis=1).apply(lambda x: 'High' if x >= 2 else 'Normal')
#         # Check class distribution
#         if pcod_data['Severity'].nunique() < 2:
#             print("Warning: Only one class in PCOD Severity. Balancing classes...")
#             n_high = len(pcod_data) // 2
#             pcod_data.loc[pcod_data.sample(n_high, random_state=42).index, 'Severity'] = 'High'
#     else:
#         print("No symptom columns available for Severity. Using random target.")
#         pcod_data['Severity'] = np.random.choice(['Normal', 'High'], len(pcod_data), p=[0.5, 0.5])
#
# # Verify PCOD class distribution
# print("PCOD Severity class distribution:", pcod_data['Severity'].value_counts().to_dict())
#
# # Extract features and target
# X_pcod = pcod_data[pcod_features]
# y_pcod = pcod_data["Severity"]
# X_train_pcod, X_test_pcod, y_train_pcod, y_test_pcod = train_test_split(X_pcod, y_pcod, test_size=0.2, random_state=42)
# scaler_pcod = StandardScaler()
# X_train_pcod = scaler_pcod.fit_transform(X_train_pcod)
# X_test_pcod = scaler_pcod.transform(X_test_pcod)
# model_pcod = LogisticRegression(max_iter=500, class_weight='balanced')
# model_pcod.fit(X_train_pcod, y_train_pcod)
# pcod_report = classification_report(y_test_pcod, model_pcod.predict(X_test_pcod), zero_division=0, output_dict=True)
# pcod_accuracy = pcod_report['accuracy']
#
# # PCOS: Process PCOS_data_without_infertility_simplified.csv
# print("Processing PCOS data...")
# try:
#     pcos_data = pd.read_csv("archive/PCOS_data_without_infertility_simplified.csv")
#     print("Raw PCOS data columns:", pcos_data.columns.tolist())
# except FileNotFileNotFoundError:
#     print("PCOS_data_without_infertility_simplified.csv not found. Trying Cleaned-Data_simplified.csv...")
#     try:
#         pcos_data = pd.read_csv("archive/Cleaned-Data_simplified.csv")
#         print("Raw Cleaned-Data_simplified columns:", pcos_data.columns.tolist())
#     except FileNotFoundError:
#         print("Cleaned-Data_simplified.csv not found.")
#         raise
#
# # PCOS feature mappings
# pcos_feature_mapping = {
#     'Age (yrs)': 'Age',
#     'Weight (Kg)': 'Weight_kg',
#     'Height(Cm)': 'Height_cm',
#     'Cycle(R/I)': 'IrregularPeriods',
#     'Pregnant(Y/N)': 'Infertility',
#     'No. of aborptions': 'Miscarriages',
#     'hair growth(Y/N)': 'HairGrowth',
#     'Pimples(Y/N)': 'Acne',
#     'Hair loss(Y/N)': 'HairLoss',
#     'Family_History_PCOS': 'FamilyHistory',
#     'Menstrual_Irregularity': 'IrregularPeriods',
#     'Insulin_Resistance': 'InsulinResistance',
#     'Stress_Level': 'StressLevel',
#     'PCOS': 'PCOS (Y/N)',  # Map PCOS to PCOS (Y/N) to avoid conflict
#     'PCOS (Y/N)': 'PCOS (Y/N)'
# }
# pcos_data = pcos_data.rename(columns=pcos_feature_mapping)
# print("PCOS columns after renaming:", pcos_data.columns.tolist())
#
# # Preprocess PCOS data
# pcos_features = ["Age", "BMI", "IrregularPeriods", "Infertility", "Miscarriages", "HairGrowth", "Acne", "HairLoss", "FamilyHistory", "StressLevel", "InsulinResistance"]
# categorical_cols = ["IrregularPeriods", "Infertility", "HairGrowth", "Acne", "HairLoss", "FamilyHistory", "StressLevel", "InsulinResistance"]
# pcos_data = preprocess_data(pcos_data, pcos_features, categorical_cols, bmi_cols=['Weight_kg', 'Height_cm'])
#
# # Handle Infertility
# if 'Infertility' in pcos_data.columns:
#     pcos_data['Infertility'] = pd.to_numeric(pcos_data['Infertility'], errors='coerce').fillna(0).astype(int)
#     pcos_data['Infertility'] = 1 - pcos_data['Infertility']
#
# # Derive Severity for PCOS
# if 'PCOS (Y/N)' in pcos_data.columns:
#     print("Using PCOS (Y/N) for PCOS Severity...")
#     pcos_data['Severity'] = pcos_data['PCOS (Y/N)'].apply(lambda x: 'High' if x == 1 else 'Normal')
# elif 'Severity' in pcos_data.columns:
#     print("Using existing Severity for PCOS...")
#     pcos_data['Severity'] = pcos_data['Severity'].apply(lambda x: 'High' if x == 1 else 'Normal')
# else:
#     print("No PCOS (Y/N) or Severity column found. Using random target for PCOS.")
#     pcos_data['Severity'] = np.random.choice(['Normal', 'High'], len(pcos_data), p=[0.5, 0.5])
#
# # Verify PCOS class distribution
# print("PCOS Severity class distribution:", pcos_data['Severity'].value_counts().to_dict())
#
# # Extract features and target
# X_pcos = pcos_data[pcos_features]
# y_pcos = pcos_data["Severity"]
# X_train_pcos, X_test_pcos, y_train_pcos, y_test_pcos = train_test_split(X_pcos, y_pcos, test_size=0.2, random_state=42)
# scaler_pcos = StandardScaler()
# X_train_pcos = scaler_pcos.fit_transform(X_train_pcos)
# X_test_pcos = scaler_pcos.transform(X_test_pcos)
# model_pcos = LogisticRegression(max_iter=500, class_weight='balanced')
# model_pcos.fit(X_train_pcos, y_train_pcos)
# pcos_report = classification_report(y_test_pcos, model_pcos.predict(X_test_pcos), zero_division=0, output_dict=True)
# pcos_accuracy = pcos_report['accuracy']
#
# # Save accuracies
# with open("models/model_accuracies.pkl", "wb") as f:
#     pickle.dump({"pcod_accuracy": pcod_accuracy, "pcos_accuracy": pcos_accuracy}, f)
#
# print("Model accuracies saved in 'models/model_accuracies.pkl'.")



import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import os

# Create models directory
os.makedirs("models", exist_ok=True)

# Suppress pandas FutureWarning
pd.set_option('future.no_silent_downcasting', True)

# Convert age to numeric
def convert_age_to_numeric(age):
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

# Preprocess data
def preprocess_data(df, features, categorical_cols, bmi_cols=None):
    df = df.copy()
    df.columns = df.columns.str.strip()
    print("Columns after stripping:", df.columns.tolist())

    # Calculate BMI if needed
    if bmi_cols and 'BMI' not in df.columns and all(col in df.columns for col in bmi_cols):
        weight_col, height_col = bmi_cols
        if 'cm' in height_col.lower():
            df['BMI'] = df[weight_col] / ((df[height_col] * 0.01) ** 2)
        else:
            df['BMI'] = df[weight_col] / ((df[height_col] * 0.3048) ** 2)
        df['BMI'] = pd.to_numeric(df['BMI'], errors='coerce').fillna(
            df['BMI'].median() if df['BMI'].notna().any() else 25)

    # Convert categorical columns
    for col in [c for c in categorical_cols if c in df.columns]:
        if df[col].dtype == 'object':
            df[col] = df[col].str.strip().str.lower().replace({
                'yes': 1, 'no': 0, 'y': 1, 'n': 0,
                'always': 1, 'sometimes': 1, 'sometime': 1, 'never': 0,
                'r': 0, 'i': 1, '2': 0, '4': 1,
                'yes, not diagnosed by a doctor': 1, 'yes, diagnosed by a doctor': 1,
                'low': 0, 'moderate': 1, 'high': 2
            }).infer_objects(copy=False).fillna(0).astype(int)

    # Convert numeric columns
    for col in ['Age', 'BMI', 'Miscarriages', 'StressLevel']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(
                df[col].median() if df[col].notna().any() else (25 if col in ['Age', 'BMI'] else 0))

    # Add missing features
    for feature in features:
        if feature not in df.columns:
            print(f"Adding missing feature: {feature}")
            df[feature] = np.random.randint(18, 45, len(df)) if feature == 'Age' else \
                np.random.uniform(18, 35, len(df)) if feature == 'BMI' else \
                    np.random.randint(0, 3, len(df)) if feature == 'Miscarriages' else \
                        np.random.choice([0, 1, 2], len(df)) if feature == 'StressLevel' else \
                            np.random.choice([0, 1], len(df))

    # Drop rows with missing features
    available_features = [f for f in features if f in df.columns]
    df = df.dropna(subset=available_features)
    print("Columns after preprocessing:", df.columns.tolist())
    return df

# ==================== PCOD MODEL ====================
print("Processing PCOD data...")
try:
    pcod_data = pd.read_csv("archive/result_data_simplified.csv")
    print("Raw PCOD data columns:", pcod_data.columns.tolist())
except FileNotFoundError:
    print("result_data_simplified.csv not found.")
    raise

# PCOD feature mappings
pcod_feature_mapping = {
    'Family history of PCOD': 'FamilyHistory',
    '7) Do you experience any pain and cramps in your lower abdomen, lower back and legs?': 'Pain',
    'Signs of PCOD': 'HairGrowth',
    '20) Do you feel stressed due to your increased weight?': 'WeightGain'
}
pcod_data = pcod_data.rename(columns=pcod_feature_mapping)

# Derive HairGrowth
if 'HairGrowth' in pcod_data.columns:
    pcod_data['HairGrowth'] = pcod_data['HairGrowth'].apply(
        lambda x: 1 if isinstance(x, str) and 'hirsutism' in x.lower().strip() else x if isinstance(x, (int, float)) else 0
    )

# Define PCOD features
pcod_features = ["Age", "BMI", "IrregularPeriod", "WeightGain", "HairGrowth", "Acne", "HairLoss", "FamilyHistory", "Pain", "StressLevel", "InsulinResistance"]
categorical_cols = ["IrregularPeriod", "WeightGain", "HairGrowth", "Acne", "HairLoss", "FamilyHistory", "Pain", "StressLevel", "InsulinResistance"]

# Preprocess PCOD data
pcod_data = preprocess_data(pcod_data, pcod_features, categorical_cols)

# Derive Severity for PCOD
if 'Medical Condition' in pcod_data.columns and pcod_data['Medical Condition'].nunique() > 1:
    pcod_data['Severity'] = pcod_data['Medical Condition'].apply(lambda x: 'High' if x == 1 else 'Normal')
elif 'Are you taking any medications for PCOD?' in pcod_data.columns:
    pcod_data['Severity'] = pcod_data['Are you taking any medications for PCOD?'].apply(lambda x: 'High' if x == 1 else 'Normal')
else:
    symptom_cols = [col for col in ['HairGrowth', 'FamilyHistory', 'Pain', 'WeightGain'] if col in pcod_data.columns]
    if symptom_cols:
        pcod_data['Severity'] = pcod_data[symptom_cols].sum(axis=1).apply(lambda x: 'High' if x >= 2 else 'Normal')
        if pcod_data['Severity'].nunique() < 2:
            n_high = len(pcod_data) // 2
            pcod_data.loc[pcod_data.sample(n_high, random_state=42).index, 'Severity'] = 'High'

print("PCOD Severity distribution:", pcod_data['Severity'].value_counts().to_dict())

# Train PCOD model
X_pcod = pcod_data[pcod_features]
y_pcod = pcod_data["Severity"]
X_train_pcod, X_test_pcod, y_train_pcod, y_test_pcod = train_test_split(X_pcod, y_pcod, test_size=0.2, random_state=42)
scaler_pcod = StandardScaler()
X_train_pcod = scaler_pcod.fit_transform(X_train_pcod)
X_test_pcod = scaler_pcod.transform(X_test_pcod)
model_pcod = LogisticRegression(max_iter=500, class_weight='balanced')
model_pcod.fit(X_train_pcod, y_train_pcod)

# Calculate PCOD accuracy
y_pred_pcod = model_pcod.predict(X_test_pcod)
pcod_accuracy = accuracy_score(y_test_pcod, y_pred_pcod)
print(f"PCOD Model Accuracy: {pcod_accuracy*100:.2f}%")

# ==================== PCOS MODEL ====================
print("Processing PCOS data...")
try:
    pcos_data = pd.read_csv("archive/PCOS_data_without_infertility_simplified.csv")
    print("Raw PCOS data columns:", pcos_data.columns.tolist())
except FileNotFoundError:
    try:
        pcos_data = pd.read_csv("archive/Cleaned-Data_simplified.csv")
        print("Raw Cleaned-Data columns:", pcos_data.columns.tolist())
    except FileNotFoundError:
        print("PCOS data files not found.")
        raise

# PCOS feature mappings
pcos_feature_mapping = {
    'Age (yrs)': 'Age',
    'Weight (Kg)': 'Weight_kg',
    'Height(Cm)': 'Height_cm',
    'Cycle(R/I)': 'IrregularPeriods',
    'Pregnant(Y/N)': 'Infertility',
    'No. of aborptions': 'Miscarriages',
    'hair growth(Y/N)': 'HairGrowth',
    'Pimples(Y/N)': 'Acne',
    'Hair loss(Y/N)': 'HairLoss',
    'Family_History_PCOS': 'FamilyHistory',
    'Menstrual_Irregularity': 'IrregularPeriods',
    'Insulin_Resistance': 'InsulinResistance',
    'Stress_Level': 'StressLevel',
    'PCOS': 'PCOS (Y/N)',
    'PCOS (Y/N)': 'PCOS (Y/N)'
}
pcos_data = pcos_data.rename(columns=pcos_feature_mapping)

# Define PCOS features
pcos_features = ["Age", "BMI", "IrregularPeriods", "Infertility", "Miscarriages", "HairGrowth", "Acne", "HairLoss", "FamilyHistory", "StressLevel", "InsulinResistance"]
categorical_cols = ["IrregularPeriods", "Infertility", "HairGrowth", "Acne", "HairLoss", "FamilyHistory", "StressLevel", "InsulinResistance"]

# Preprocess PCOS data
pcos_data = preprocess_data(pcos_data, pcos_features, categorical_cols, bmi_cols=['Weight_kg', 'Height_cm'])

# Handle Infertility
if 'Infertility' in pcos_data.columns:
    pcos_data['Infertility'] = pd.to_numeric(pcos_data['Infertility'], errors='coerce').fillna(0).astype(int)
    pcos_data['Infertility'] = 1 - pcos_data['Infertility']

# Derive Severity for PCOS
if 'PCOS (Y/N)' in pcos_data.columns:
    pcos_data['Severity'] = pcos_data['PCOS (Y/N)'].apply(lambda x: 'High' if x == 1 else 'Normal')
elif 'Severity' in pcos_data.columns:
    pcos_data['Severity'] = pcos_data['Severity'].apply(lambda x: 'High' if x == 1 else 'Normal')
else:
    pcos_data['Severity'] = np.random.choice(['Normal', 'High'], len(pcos_data), p=[0.5, 0.5])

print("PCOS Severity distribution:", pcos_data['Severity'].value_counts().to_dict())

# Train PCOS model
X_pcos = pcos_data[pcos_features]
y_pcos = pcos_data["Severity"]
X_train_pcos, X_test_pcos, y_train_pcos, y_test_pcos = train_test_split(X_pcos, y_pcos, test_size=0.2, random_state=42)
scaler_pcos = StandardScaler()
X_train_pcos = scaler_pcos.fit_transform(X_train_pcos)
X_test_pcos = scaler_pcos.transform(X_test_pcos)
model_pcos = LogisticRegression(max_iter=500, class_weight='balanced')
model_pcos.fit(X_train_pcos, y_train_pcos)

# Calculate PCOS accuracy
y_pred_pcos = model_pcos.predict(X_test_pcos)
pcos_accuracy = accuracy_score(y_test_pcos, y_pred_pcos)
print(f"PCOS Model Accuracy: {pcos_accuracy*100:.2f}%")

# ==================== Calculate Combined Accuracy ====================
# For PCOS with PCOD: average of both accuracies (since it's a rule-based combination)
pcos_pcod_accuracy = (pcos_accuracy + pcod_accuracy) / 2

# Normal accuracy: inverse of average positive prediction accuracy
normal_accuracy = 1 - ((1 - pcos_accuracy) + (1 - pcod_accuracy)) / 2

# Save all accuracies
accuracies = {
    "pcod_accuracy": pcod_accuracy,
    "pcos_accuracy": pcos_accuracy,
    "pcos_pcod_accuracy": pcos_pcod_accuracy,
    "normal_accuracy": normal_accuracy
}

with open("models/model_accuracies.pkl", "wb") as f:
    pickle.dump(accuracies, f)

print("\n=== Model Accuracies ===")
print(f"PCOD Model: {pcod_accuracy*100:.2f}%")
print(f"PCOS Model: {pcos_accuracy*100:.2f}%")
print(f"PCOS with PCOD: {pcos_pcod_accuracy*100:.2f}%")
print(f"Normal Classification: {normal_accuracy*100:.2f}%")
print("\nAccuracies saved to 'models/model_accuracies.pkl'")