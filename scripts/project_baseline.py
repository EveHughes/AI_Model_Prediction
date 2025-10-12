"""
This Python file provides some useful code for reading the training file
"clean_dataset.csv". You may adapt this code as you see fit. However,
keep in mind that the code provided does only basic feature transformations
to build a rudimentary kNN model in sklearn. Not all features are considered
in this code, and you should consider those features! Use this code
where appropriate, but don't stop here!
"""

import numpy as np
import pandas as pd
import re
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MultiLabelBinarizer


file_name = "data/training_data_clean.csv"


def process_multiselect(series, target_tasks):
    """Convert multiselect strings to lists, keeping only specified features"""
    processed = []
    for response in series:
        if pd.isna(response) or response == '':
            processed.append([])
        else:
            # Check which of the target tasks are present in the response
            present_tasks = [task for task in target_tasks if task in str(response)]
            processed.append(present_tasks)
    return processed


def extract_rating(response):
    """
    Extract numeric rating from responses like '3 - Sometimes'.
    Returns None for missing responses
    """
    match = re.match(r'^(\d+)', str(response))
    return int(match.group(1)) if match else None


def main():
    # Load processed data
    df = pd.read_csv(file_name)

    # Drop rows with missing data
    df.dropna(inplace=True)

    # Define the tasks we want to use as features (first four, clean ones)
    target_tasks = [
        'Math computations',
        'Writing or debugging code',
        'Data processing or analysis', 
        'Explaining complex concepts simply',
    ]

    # Process multi-select columns (use exact column names from your cleaned data)
    best_tasks_lists = process_multiselect(df['Which types of tasks do you feel this model handles best? (Select all that apply.)'], target_tasks)
    suboptimal_tasks_lists = process_multiselect(df['For which types of tasks do you feel this model tends to give suboptimal responses? (Select all that apply.)'], target_tasks)
    
    mlb_best = MultiLabelBinarizer()
    mlb_subopt = MultiLabelBinarizer()
    
    best_tasks_encoded = mlb_best.fit_transform(best_tasks_lists)
    suboptimal_tasks_encoded = mlb_subopt.fit_transform(suboptimal_tasks_lists)

    # Use some rating features
    academic_numeric = df['How likely are you to use this model for academic tasks?'].apply(extract_rating)
    subopt_numeric = df['Based on your experience, how often has this model given you a response that felt suboptimal?'].apply(extract_rating)

    # Combine features
    X = np.hstack([academic_numeric.values.reshape(-1, 1), subopt_numeric.values.reshape(-1, 1), 
                   best_tasks_encoded, suboptimal_tasks_encoded])
    y = df['label'].values

    # Simple train / test split
    n_train = int(0.7 * len(X))
    X_train, y_train, X_test, y_test = X[:n_train], y[:n_train], X[n_train:], y[n_train:]

    # Train simple KNN
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)

    # Evaluate
    train_acc = knn.score(X_train, y_train)
    test_acc = knn.score(X_test, y_test)

    print(f"Training accuracy: {train_acc:.3f}")
    print(f"Test accuracy: {test_acc:.3f}")

if __name__ == "__main__":
    main()