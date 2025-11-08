import pandas as pd
import math

def make_bow(text_data_pairs, vocab):
    """
    Produce the bag-of-word representation of the text data, along with a vector
    of labels.
    """
    vocab_map = {word: j for j, word in enumerate(vocab)}
    X_list = []
    t_list = []
    
    for response, label in text_data_pairs:
        bow_vector = [0] * len(vocab)
        words = response.split(' ') 
        unique_words = set(word.strip().lower() for word in words)
        
        for word in unique_words:
            if word in vocab_map:
                j = vocab_map[word]
                bow_vector[j] = 1
        X_list.append(bow_vector)
        
        # Map labels to integers
        label_lower = label.lower()
        if label_lower == 'chatgpt': 
            t_list.append(0)
        elif label_lower == 'claude':
            t_list.append(1)
        elif label_lower == 'gemini':
            t_list.append(2)
        else:
            # Should not happen if data is clean, but skip for safety
            continue 
            
    X = pd.DataFrame(X_list, columns=vocab)
    t = pd.Series(t_list, dtype='int64')
    
    X = X.reset_index(drop=True)
    t = t.reset_index(drop=True)
    
    return X, t 

def create_features(df, text_columns, label_col_name, vocab_list=None, ohe_encoders=None):
    """
    Creates a full feature matrix X (BoW + OHE + Binary) and labels t.
    If vocab_list or ohe_encoders is None, they are built from the dataframe (training mode).
    """
    
    df_proc = df.copy()
    for col in text_columns:
        df_proc[col] = df_proc[col].fillna('')
        
    ordinal_columns = [
        'How likely are you to use this model for academic tasks?', 
        'Based on your experience, how often has this model given you a response that felt suboptimal?',
        'How often do you expect this model to provide responses with references or supporting evidence?', 
        'How often do you verify this model\'s responses?'
    ]
    for col in ordinal_columns:
        mode_val = df_proc[col].mode()[0] if not df_proc[col].mode().empty else 1 
        df_proc[col] = df_proc[col].fillna(mode_val)
        
    # Create the concatenated text pairs for BoW 
    text_data_pairs = []
    for index, row in df_proc.iterrows():
        full_text = ' '.join(row[col].lower().strip() for col in text_columns)
        label = row[label_col_name]
        text_data_pairs.append((full_text, label))

    is_training = vocab_list is None
    
    if is_training:
        # Build vocabulary from text in the training data
        unique_vocab = set()
        for text, _ in text_data_pairs:
            unique_vocab.update(text.split(' '))
        vocab_list = sorted(list(unique_vocab))
        
    # Generate BoW matrix and encoded labels (t)
    X_text, t = make_bow(text_data_pairs, vocab_list)
    
    if X_text.empty:
        return pd.DataFrame(), pd.Series(), vocab_list, ohe_encoders
    df_proc_aligned = df_proc.iloc[t.index].reset_index(drop=True)

    if is_training:
        ohe_encoders = {}
        X_ordinal_list = []
        for col in ordinal_columns:
            ohe_map = {val: f"{col}={val}" for val in df_proc_aligned[col].unique()}
            ohe_encoders[col] = ohe_map
            
            ohe_df = pd.DataFrame(0, index=df_proc_aligned.index, columns=ohe_map.values())
            for idx, val in df_proc_aligned[col].items():
                ohe_df.loc[idx, ohe_map[val]] = 1
            X_ordinal_list.append(ohe_df)
    else:
        X_ordinal_list = []
        for col in ordinal_columns:
            ohe_map = ohe_encoders[col]
            ohe_df = pd.DataFrame(0, index=df_proc_aligned.index, columns=ohe_map.values())
            
            def apply_ohe(val):
                if val in ohe_map:
                    return {ohe_map[val]: 1}
                return {}
            
            ohe_rows = df_proc_aligned[col].apply(apply_ohe).tolist()
            
            for idx, row_dict in enumerate(ohe_rows):
                for ohe_col, val in row_dict.items():
                    ohe_df.loc[idx, ohe_col] = val
            
            X_ordinal_list.append(ohe_df)

    X_ordinal = pd.concat(X_ordinal_list, axis=1)

    binary_columns = [col for col in df_proc.columns if col.startswith(('best_task_', 'subopt_task_'))]
    X_binary = df_proc_aligned[binary_columns].reset_index(drop=True)

    X = pd.concat([X_text, X_ordinal, X_binary], axis=1)

    if is_training:
        return X, t, vocab_list, ohe_encoders
    else:
        return X, t, vocab_list, ohe_encoders

class NaiveBayes:
    """
    A Bernoulli Naive Bayes classifier.
    This model calculates predictions based on the log-likelihood of
    features and the log-prior probability of each class.
    """
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.log_likelihoods_ = None
        self.log_neg_likelihoods_ = None
        self.log_priors_ = None 
        self.classes_ = None

    def fit(self, X, t):
        self.classes_ = sorted(t.unique())
        n_samples = len(t)
        
        # Calculate Log Priors: log(P(c))
        class_counts = t.value_counts().sort_index() 
        self.log_priors_ = (class_counts / n_samples).apply(math.log)
        
        # log(P(word | c))
        feature_counts = X.groupby(t).sum()
        
        # Apply Laplace Smoothing for Bernoulli NB: P(w|c) = (count(w,c) + alpha) / (count_docs_in_c + 2 * alpha)
        numerator = feature_counts + self.alpha
        denominator_series = class_counts.apply(lambda cc: cc + 2 * self.alpha) 
        
        probs = numerator.div(denominator_series, axis=0)
        
        # Handle potential log(0)
        probs = probs.clip(1e-14, 1.0 - 1e-14) 

        self.log_likelihoods_ = probs.map(math.log)
        self.log_neg_likelihoods_ = (1 - probs).map(math.log)
        
    def predict(self, X):
        if self.log_likelihoods_ is None:
            raise ValueError("Model has not been trained. Call fit() first.")
            
        posteriors_list = []
        
        for c in self.classes_:
            train_cols = self.log_likelihoods_.columns.intersection(X.columns)
            
            ll = self.log_likelihoods_.loc[c, train_cols] 
            nll = self.log_neg_likelihoods_.loc[c, train_cols] 
            X_aligned = X[train_cols]
            
            log_prior_c = self.log_priors_.loc[c] 
            log_likelihood_score = X_aligned.dot(ll) + (1 - X_aligned).dot(nll)
            log_score = log_likelihood_score + log_prior_c
            
            log_score.name = c 
            posteriors_list.append(log_score)
            
        scores = pd.concat(posteriors_list, axis=1)
        return scores.idxmax(axis=1)


if __name__ == "__main__":
    k = 5
    alpha = 1.0 
    trainfile = 'data/train_data.csv' 
    
    text_columns = [
        "In your own words, what kinds of tasks would you use this model for?", 
        "Think of one task where this model gave you a suboptimal response. What did the response look like, and why did you find it suboptimal?",
        "When you verify a response from this model, how do you usually go about it?"
    ]
    label_col_name = 'label'

    print(f"Loading all data for {k}-fold cross-validation")
    try:
        df = pd.read_csv(trainfile).drop(columns=['student_id'])
    except FileNotFoundError:
        print(f"Error: The file '{trainfile}' was not found.")
        exit()
        
    if df.empty:
        print("Stopping: No data was loaded.")
        exit()
        
    indices = df.index.tolist()
    fold_indices = []
    fold_size_base = len(indices) // k
    remainder = len(indices) % k
    
    start_index = 0
    for i in range(k):
        current_fold_size = fold_size_base + (1 if i < remainder else 0)
        end_index = start_index + current_fold_size
        
        fold_indices.append(indices[start_index:end_index])
        start_index = end_index

    print(f"Data split into {k} folds. (Sizes: {[len(f) for f in fold_indices]})")

    fold_accuracies = []

    for i in range(k):
        print(f"\nFold {i+1}/{k}")
        
        # Get training and validation row indices
        val_indices = fold_indices[i]
        train_indices = [idx for j, f_indices in enumerate(fold_indices) if i != j for idx in f_indices]
        
        # Create training and validation DataFrames
        df_train = df.loc[train_indices].reset_index(drop=True)
        df_val = df.loc[val_indices].reset_index(drop=True)
        
        print(f"Training samples: {len(df_train)}, Validation samples: {len(df_val)}")

        X_train, t_train, vocab_list, ohe_encoders = create_features(
            df_train, text_columns, label_col_name, vocab_list=None, ohe_encoders=None
        )
        print(f"Vocabulary size for this fold: {len(vocab_list)}")

        X_val, t_val, _, _ = create_features(
            df_val, text_columns, label_col_name, vocab_list=vocab_list, ohe_encoders=ohe_encoders
        )
        
        if X_train.empty or X_val.empty:
             print("Skipping fold due to empty feature matrix.")
             continue

        # Train model
        model = NaiveBayes(alpha=alpha)
        model.fit(X_train, t_train)

        # Predict and calculate accuracy
        t_pred = model.predict(X_val)
        
        correct = (t_pred == t_val).sum()
        total = len(t_val)
        accuracy = correct / total if total > 0 else 0
        
        fold_accuracies.append(accuracy)
        print(f"Fold {i+1} Accuracy: {accuracy * 100:.2f}%")

    if fold_accuracies:
        mean_accuracy = sum(fold_accuracies) / len(fold_accuracies)
        print("\nCross Validation Done")
        print(f"Accuracies per fold: {[f'{acc*100:.2f}%' for acc in fold_accuracies]}")
        print(f"Mean Accuracy (k={k}): {mean_accuracy * 100:.2f}%")
    else:
        print("\nNo folds were processed")