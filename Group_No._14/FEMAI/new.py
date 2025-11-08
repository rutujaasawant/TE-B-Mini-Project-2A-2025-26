import pandas as pd
import numpy as np

def binarize_col(series, yes_vals=['yes', 'y', 'always', 'sometimes', 'sometime'], no_vals=['no', 'n', 'never']):
    return series.str.lower().map(lambda x: 1 if x in yes_vals else 0 if x in no_vals else np.nan)

def simplify_result_data(csvfile, outfile):
    df = pd.read_csv(csvfile)
    # Drop personal text fields
    drop_cols = [c for c in df.columns if any(x in c.lower() for x in ('name', 'email', 'timestamp'))]
    df = df.drop(columns=drop_cols)
    # Convert categorical columns to binary
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = binarize_col(df[col].astype(str).fillna('no'))
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    df.to_csv(outfile, index=False)

def simplify_cleaned_data(csvfile, outfile):
    df = pd.read_csv(csvfile)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = binarize_col(df[col].astype(str).fillna('no'))
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    df.to_csv(outfile, index=False)

def simplify_pcos_xlsx(xlsxfile, outfile):
    sheets = pd.ExcelFile(xlsxfile).sheet_names
    df = pd.read_excel(xlsxfile, sheet_name=sheets[1] if "Full_new" in sheets else sheets[0])
    drop_cols = [c for c in df.columns if any(x in c.lower() for x in ('name', 'pcos', 'file', 'patient'))]
    df = df.drop(columns=drop_cols, errors='ignore')
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = binarize_col(df[col].astype(str).fillna('no'))
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    df.to_csv(outfile, index=False)

# Provide file paths for input and output
simplify_result_data('archive/result_data.csv', 'result_data_simplified.csv')
simplify_cleaned_data('archive/Cleaned-Data.csv', 'Cleaned-Data_simplified.csv')
simplify_pcos_xlsx('PCOS_data_without_infertility.xlsx', 'PCOS_data_without_infertility_simplified.csv')
