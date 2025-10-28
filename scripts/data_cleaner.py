import numpy as np
import pandas as pd
import os

file_name = "../data/training_data_clean.csv"

if not os.path.exists(file_name):
    print(f"Error: File not found at {file_name}")
    print("Please check the file path.")
else:
    data = pd.read_csv(file_name)
    print(f"Loaded {file_name} successfully.")

    to_int = ["How likely are you to use this model for academic tasks?", "Based on your experience, how often has this model given you a response that felt suboptimal?", "How often do you expect this model to provide responses with references or supporting evidence?", "How often do you verify this model's responses?"]

    for col in to_int:
        if col in data.columns:
            # Convert to string, strip whitespace, get first character, convert to numeric
            data[col] = pd.to_numeric(data[col].astype(str).str.strip().str[0], errors='coerce').astype('Int64')
            print(f"Converted column to numeric: {col}")
        else:
            print(f"Warning: expected column {col!r} not found in data")

    print("Cleaning object columns (stripping whitespace, replacing '#NAME?' and empty strings)...")
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = data[col].str.strip()
            # Replace '#NAME?' and empty strings with NA
            data[col] = data[col].replace({'#NAME?': pd.NA, '': pd.NA})

    out_file = '../data/cleaned_data.csv'
    out_dir = os.path.dirname(out_file) or '.'
    os.makedirs(out_dir, exist_ok=True)
    
    data.to_csv(out_file, index=False)
    
    print(f"\nData cleaning complete. Cleaned data saved to: {out_file}")
