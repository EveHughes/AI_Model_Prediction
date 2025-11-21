import numpy as np
import pandas as pd
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
import pickle


def process_multiselect(series, target_tasks):
    """Convert multiselect strings to lists, keeping only specified features"""
    processed = []
    for response in series:
        if pd.isna(response) or response == '' or response == '#NAME?':
            processed.append([])
        else:
            present_tasks = [task for task in target_tasks if task in str(response)]
            processed.append(present_tasks)
    return processed


def extract_rating(response):
    """Extract numeric rating from responses like '3 — Sometimes'"""
    if pd.isna(response) or response == '' or response == '#NAME?':
        return 0  # Default value for missing
    match = re.match(r'^(\d+)', str(response))
    return int(match.group(1)) if match else 0


def process_text_features(series):
    """Process text responses, replacing missing values"""
    return series.fillna('').replace('#NAME?', '').astype(str)


def main():
    print("Loading data...")
    df = pd.read_csv("data/training_data_clean.csv")
    
    # Replace missing values and #NAME? with empty strings
    df = df.fillna('')
    df = df.replace('#NAME?', '')
    
    print(f"Total data points: {len(df)}")
    print(f"Class distribution:\n{df['label'].value_counts()}")
    
    # Define target tasks for multi-select features
    target_tasks = [
        'Math computations',
        'Writing or debugging code',
        'Data processing or analysis', 
        'Explaining complex concepts simply',
    ]
    
    # Process multi-select columns(using exact column names)
    print("\nProcessing multi-select features...")
    best_tasks_lists = process_multiselect(
        df['Which types of tasks do you feel this model handles best? (Select all that apply.)'], 
        target_tasks
    )
    suboptimal_tasks_lists = process_multiselect(
        df['For which types of tasks do you feel this model tends to give suboptimal responses? (Select all that apply.)'], 
        target_tasks
    )
    
    mlb_best = MultiLabelBinarizer()
    mlb_subopt = MultiLabelBinarizer()
    
    best_tasks_encoded = mlb_best.fit_transform(best_tasks_lists)
    suboptimal_tasks_encoded = mlb_subopt.fit_transform(suboptimal_tasks_lists)
    
    # Extract numerical ratings - using exact column names
    print("Processing rating features...")
    academic_numeric = df['How likely are you to use this model for academic tasks?'].apply(extract_rating)
    subopt_numeric = df['Based on your experience, how often has this model given you a response that felt suboptimal?'].apply(extract_rating)
    references_numeric = df['How often do you expect this model to provide responses with references or supporting evidence?'].apply(extract_rating)
    verify_numeric = df["How often do you verify this model's responses?"].apply(extract_rating)
    
    # Process text features - there are 3 text columns in your data
    print("Processing text features...")
    text_columns = [
        'In your own words, what kinds of tasks would you use this model for?',
        'Think of one task where this model gave you a suboptimal response. What did the response look like, and why did you find it suboptimal?',
        'When you verify a response from this model, how do you usually go about it?'
    ]
    
    # Combine all text features
    combined_text = df[text_columns].apply(
        lambda row: ' '.join(process_text_features(row)), axis=1
    )
    
    # Use TF-IDF for text vectorization
    tfidf = TfidfVectorizer(max_features=100, stop_words='english', min_df=2, max_df=0.8)
    text_features = tfidf.fit_transform(combined_text).toarray()
    
    # Combine all features
    print("Combining features...")
    X = np.hstack([
        academic_numeric.values.reshape(-1, 1),
        subopt_numeric.values.reshape(-1, 1),
        references_numeric.values.reshape(-1, 1),
        verify_numeric.values.reshape(-1, 1),
        best_tasks_encoded,
        suboptimal_tasks_encoded,
        text_features
    ])
    
    y = df['label'].values
    
    print(f"Feature matrix shape: {X.shape}")
    
    # Split data: rows 0-743 for training/validation, rows 744-824 for test
    # so 744 training samples and 81 test samples
    X_train = X[:744]
    y_train = y[:744]
    X_test = X[744:]
    y_test = y[744:]
    
    print(f"\nTraining/Validation set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # defined hyperparameter grid for Random Forest
    print("\nSetting up Grid Search...")
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', None],
        'bootstrap': [True]
    }
    
    # Create Random Forest classifier
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    # Grid Search with CV
    # Total combinations: 3 × 2 × 2 × 2 × 2 × 1 = 48 (look at the param_Grid)
    # With 3-fold CV: 48 × 3 = 144 model trainings
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=3,  # 3-fold cross-validation (faster than 5-fold, still reliable)
        scoring='accuracy',
        n_jobs=-1,
        verbose=2
    )
    
    print("Starting Grid Search (this may take a while)...")
    grid_search.fit(X_train, y_train)
    
    # best model
    best_rf = grid_search.best_estimator_
    
    print("\n" + "="*60)
    print("BEST HYPERPARAMETERS:")
    print("="*60)
    for param, value in grid_search.best_params_.items():
        print(f"{param}: {value}")
    
    # Evaluate on training set
    train_pred = best_rf.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)
    
    # Evaluate on test set
    test_pred = best_rf.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    test_f1_micro = f1_score(y_test, test_pred, average='micro')
    
    print("\n" + "="*60)
    print("RESULTS:")
    print("="*60)
    print(f"Best CV Score (Validation Accuracy): {grid_search.best_score_:.4f}")
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1-Score (Micro): {test_f1_micro:.4f}")
    
    model_data = {
        'model': best_rf,
        'mlb_best': mlb_best,
        'mlb_subopt': mlb_subopt,
        'tfidf': tfidf,
        'target_tasks': target_tasks,
        'text_columns': text_columns
    }
    
    with open('random_forest_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print("Model saved as 'random_forest_model.pkl'")
    print("\nTraining complete!")


if __name__ == "__main__":
    main()