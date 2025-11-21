import os
import pandas as pd
import numpy as np
import json
from data_cleaner import clean

def make_bow(text_data_pairs, vocab):
    """
    Produce the bag-of-word representation of the text data.
    Ensures X and t have aligned, default indices.
    """
    vocab_map = {word: j for j, word in enumerate(vocab)}
    X_list = []
    t_list = []
    
    label_map = {'chatgpt': 0, 'claude': 1, 'gemini': 2}
    
    for response, label in text_data_pairs:
        bow_vector = np.zeros(len(vocab), dtype=np.int8) 
        words = response.split(' ') 
        unique_words = set(word.strip().lower() for word in words)
        
        for word in unique_words:
            if word in vocab_map:
                j = vocab_map[word]
                bow_vector[j] = 1
        
        X_list.append(bow_vector)
        
        if isinstance(label, str) and label.lower() in label_map:
            t_list.append(label_map[label.lower()])
        else:
            t_list.append(-1)
            
    if X_list:
        X_np = np.stack(X_list)
        X = pd.DataFrame(X_np, columns=vocab)
        t = pd.Series(t_list, dtype='int64')
    else:
        X = pd.DataFrame()
        t = pd.Series(dtype='int64')
    
    return X.reset_index(drop=True), t.reset_index(drop=True) 

def create_features(df, text_columns, label_col_name, vocab_list=None, ohe_encoders=None):
    """
    Creates a full feature matrix X. 
    Crucial: Resets index of input df to align with generated vectors.
    """
    df_proc = df.copy().reset_index(drop=True)
    
    for col in text_columns:
        if col not in df_proc.columns:
             df_proc[col] = ''
        df_proc[col] = df_proc[col].fillna('')
        
    ordinal_columns = [
        'How likely are you to use this model for academic tasks?', 
        'Based on your experience, how often has this model given you a response that felt suboptimal?',
        'How often do you expect this model to provide responses with references or supporting evidence?', 
        'How often do you verify this model\'s responses?'
    ]
    for col in ordinal_columns:
        if col not in df_proc.columns:
            df_proc[col] = 1
        else:
            mode_val = df_proc[col].mode()[0] if not df_proc[col].mode().empty else 1 
            df_proc[col] = df_proc[col].fillna(mode_val)
        
    text_data_pairs = []
    for index, row in df_proc.iterrows():
        full_text = ' '.join(row[col].lower().strip() for col in text_columns if col in row)
        
        if label_col_name in df_proc.columns:
            label = row[label_col_name]
        else:
            label = "unknown" # Dummy label for test data
            
        text_data_pairs.append((full_text, label))

    is_training = vocab_list is None
    
    if is_training:
        unique_vocab = set()
        for text, _ in text_data_pairs:
            unique_vocab.update(text.split(' '))
        vocab_list = sorted(list(unique_vocab))
        
    X_text, t = make_bow(text_data_pairs, vocab_list)
    
    if X_text.empty:
        return pd.DataFrame(), pd.Series(), vocab_list, ohe_encoders
    
    df_proc_aligned = df_proc.loc[t.index].reset_index(drop=True)

    if is_training:
        ohe_encoders = {}
        X_ordinal_list = []
        for col in ordinal_columns:
            ohe_map = {val: f"{col}={val}" for val in df_proc_aligned[col].unique()}
            ohe_encoders[col] = ohe_map
            
            ohe_df = pd.get_dummies(df_proc_aligned[col], prefix=col)
            ohe_df.columns = [f"{col}={c}" for c in ohe_df.columns.str.replace(f"{col}_", "")]
            ohe_df = ohe_df[[c for c in ohe_map.values() if c in ohe_df.columns]]
            X_ordinal_list.append(ohe_df)
    else:
        X_ordinal_list = []
        for col in ordinal_columns:
            if col in ohe_encoders:
                ohe_map = ohe_encoders[col]
                ohe_df_temp = pd.get_dummies(df_proc_aligned[col], prefix=col)
                ohe_df_temp.columns = [f"{col}={c}" for c in ohe_df_temp.columns.str.replace(f"{col}_", "")]
                
                ohe_df = pd.DataFrame(0, index=df_proc_aligned.index, columns=ohe_map.values())
                
                common_cols = ohe_df.columns.intersection(ohe_df_temp.columns)
                ohe_df[common_cols] = ohe_df_temp[common_cols]
                
                X_ordinal_list.append(ohe_df)

    X_ordinal = pd.concat(X_ordinal_list, axis=1) if X_ordinal_list else pd.DataFrame()

    binary_columns = [col for col in df_proc.columns if col.startswith(('best_task_', 'subopt_task_'))]
    X_binary = df_proc_aligned[binary_columns].reset_index(drop=True)

    X = pd.concat([X_text, X_ordinal, X_binary], axis=1)

    return X, t, vocab_list, ohe_encoders

class NaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.log_likelihoods_ = None
        self.log_neg_likelihoods_ = None
        self.log_priors_ = None 
        self.classes_ = None

    def fit(self, X, t):
        valid_mask = t >= 0
        X = X[valid_mask]
        t = t[valid_mask]

        self.classes_ = sorted(t.unique())
        n_samples = len(t)
        
        class_counts = t.value_counts().sort_index() 
        self.log_priors_ = np.log(class_counts.values / n_samples) 
        self.log_priors_ = pd.Series(self.log_priors_, index=class_counts.index)
        
        feature_counts = X.groupby(t).sum() 
        
        numerator = feature_counts.values + self.alpha
        denominator_np = class_counts.values[:, np.newaxis] + 2 * self.alpha 
        
        probs = numerator / denominator_np
        probs = np.clip(probs, 1e-14, 1.0 - 1e-14) 

        self.log_likelihoods_ = np.log(probs)
        self.log_neg_likelihoods_ = np.log(1.0 - probs)
        
        self.log_likelihoods_ = pd.DataFrame(self.log_likelihoods_, index=feature_counts.index, columns=feature_counts.columns)
        self.log_neg_likelihoods_ = pd.DataFrame(self.log_neg_likelihoods_, index=feature_counts.index, columns=feature_counts.columns)
        
    def predict(self, X):
        if self.log_likelihoods_ is None:
            raise ValueError("Model has not been trained.")
            
        posteriors_list = []
        train_cols = self.log_likelihoods_.columns.intersection(X.columns)
        X_aligned = X[train_cols].values 
        
        for c in self.classes_:
            ll = self.log_likelihoods_.loc[c, train_cols].values
            nll = self.log_neg_likelihoods_.loc[c, train_cols].values
            
            log_prior_c = self.log_priors_.loc[c] 
            
            log_likelihood_score = X_aligned.dot(ll) + (1 - X_aligned).dot(nll)
            log_score_np = log_likelihood_score + log_prior_c
            
            log_score = pd.Series(log_score_np, index=X.index, name=c)
            posteriors_list.append(log_score)
            
        scores = pd.concat(posteriors_list, axis=1)
        return scores.idxmax(axis=1)

def predict_all(filename):
    # Load weights
    with open('naive_bayes_weights.json', 'r') as f:
        weights = json.load(f)
        
    model = NaiveBayes(alpha=weights['alpha'])
    model.classes_ = sorted(weights['classes'])
    
    priors_dict = {int(k): v for k, v in weights['log_priors'].items()}
    model.log_priors_ = pd.Series(priors_dict).sort_index()
    
    ll = weights['log_likelihoods']
    model.log_likelihoods_ = pd.DataFrame(ll['data'], index=ll['index'], columns=ll['columns'])
    
    nll = weights['log_neg_likelihoods']
    model.log_neg_likelihoods_ = pd.DataFrame(nll['data'], index=nll['index'], columns=nll['columns'])
    
    ohe_encoders = weights['ohe_encoders']
    
    all_columns = set(model.log_likelihoods_.columns)
    
    ordinal_features = set()
    for encoder in ohe_encoders.values():
        ordinal_features.update(encoder.values())
        
    binary_features = {col for col in all_columns if col.startswith('best_task_') or col.startswith('subopt_task_')}
    
    vocab = sorted(list(all_columns - ordinal_features - binary_features))

    # Process the test file
    clean(filename)
    
    test_path = "testing_data.csv"
    if not os.path.exists(test_path):
        return []

    df_test = pd.read_csv(test_path)
    if 'student_id' in df_test.columns:
        df_test = df_test.drop(columns=['student_id'])

    text_columns = [
        "In your own words, what kinds of tasks would you use this model for?", 
        "Think of one task where this model gave you a suboptimal response. What did the response look like, and why did you find it suboptimal?",
        "When you verify a response from this model, how do you usually go about it?"
    ]
    label_col_name = 'label'

    X_test, _, _, _ = create_features(
        df_test, text_columns, label_col_name, 
        vocab_list=vocab,
        ohe_encoders=ohe_encoders 
    )

    if X_test.empty:
        return []

    predictions_numeric = model.predict(X_test)
    
    reverse_label_map = {0: 'ChatGPT', 1: 'Claude', 2: 'Gemini'}
    predictions_list = predictions_numeric.map(reverse_label_map).tolist()

    return predictions_list