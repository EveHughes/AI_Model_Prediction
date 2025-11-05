import numpy as np
import pandas as pd
import os

if __name__ == "__main__":

    file_name = "data/training_data_clean.csv"

    if not os.path.exists(file_name):
        print(f"Error: File not found at {file_name}")
        print("Please check the file path.")
    else:
        data = pd.read_csv(file_name)
        print(f"Loaded {file_name} successfully.")

        # Convert to integers
        to_int = ["How likely are you to use this model for academic tasks?", "Based on your experience, how often has this model given you a response that felt suboptimal?", "How often do you expect this model to provide responses with references or supporting evidence?", "How often do you verify this model's responses?"]

        for col in to_int:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col].astype(str).str.strip().str[0], errors='coerce').astype('Int64')
                print(f"Converted column to numeric: {col}")
            else:
                print(f"Warning: expected column {col!r} not found in data")

        # Vectorize Select All columns
        multi_select_cols_map = {
            "Which types of tasks do you feel this model handles best? (Select all that apply.)": "best_task",
            "For which types of tasks do you feel this model tends to give suboptimal responses? (Select all that apply.)": "subopt_task"
        }
        
        separator = ','
        all_new_cols_df_list = [] # To store all dummy vectors

        for col_name, prefix in multi_select_cols_map.items():
            if col_name in data.columns:
                print(f"Vectorizing multi-select column: {col_name}")
                
                list_series = data[col_name].fillna('').str.split(separator)
                dummies_df = pd.get_dummies(list_series.explode().str.strip()).groupby(level=0).sum()

                # Remove the empty string column 
                if '' in dummies_df.columns:
                    dummies_df = dummies_df.drop(columns=[''])
                
                new_col_names = {}
                for dummy_col in dummies_df.columns:
                    clean_name = dummy_col.lower().replace(' ', '_').replace('(', '').replace(')', '')
                    new_col_names[dummy_col] = f"{prefix}_{clean_name}"
                
                dummies_df = dummies_df.rename(columns=new_col_names)
                all_new_cols_df_list.append(dummies_df)
            else:
                 print(f"Warning: expected multi-select column {col_name!r} not found in data")

        if all_new_cols_df_list:
            data = pd.concat([data] + all_new_cols_df_list, axis=1)
            # Drop the original text columns as they are now vectorized
            data = data.drop(columns=multi_select_cols_map.keys(), errors='ignore')

        # Clean remaining object columns
        print("Cleaning object columns (stripping whitespace, replacing '#NAME?' and empty strings")
        for col in data.columns:
            if data[col].dtype == 'object':
                data[col] = data[col].str.strip()
                data[col] = data[col].replace({'#NAME?': pd.NA, '': pd.NA})

        # Save Data
        out_file = 'data/cleaned_data.csv'
        out_dir = os.path.dirname(out_file) or '.'
        os.makedirs(out_dir, exist_ok=True)
        
        data.to_csv(out_file, index=False)
        
        print(f"\nData cleaning and vectorization complete. Cleaned data saved to: {out_file}")