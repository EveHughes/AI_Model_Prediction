from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
import json
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def make_bow(text_data_pairs, vocab):
    """
    Produce the bag-of-word representation of the text data, along with a vector
    of labels, using NumPy for the feature matrix.
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
                bow_vector[j] = 1 # feature indicator
        
        X_list.append(bow_vector)
        
        label_lower = label.lower()
        if label_lower in label_map:
            t_list.append(label_map[label_lower])
        else:
            # Should not happen if data is clean, but skip for safety
            continue
            
    # Convert list of NumPy arrays to a single DataFrame and Series
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
        if col in df_proc.columns:
            mode_val = df_proc[col].mode()[0] if not df_proc[col].mode().empty else 1 
            df_proc[col] = df_proc[col].fillna(mode_val)
        else:
            df_proc[col] = 1
        
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
    # df_proc must align with the rows generated in make_bow 
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
    
    Uses NumPy for vectorized math operations.
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
        self.log_priors_ = np.log(class_counts.values / n_samples) 
        self.log_priors_ = pd.Series(self.log_priors_, index=class_counts.index)
        
        # log(P(word | c))
        feature_counts = X.groupby(t).sum() 
        
        # Apply Laplace Smoothing for Bernoulli NB: P(w|c) = (count(w,c) + alpha) / (count_docs_in_c + 2 * alpha)
        numerator = feature_counts.values + self.alpha
        denominator_np = class_counts.values[:, np.newaxis] + 2 * self.alpha 
        
        # Vectorized probability calculation
        probs = numerator / denominator_np
        
        # Handle potential log(0) using numpy.clip
        probs = np.clip(probs, 1e-14, 1.0 - 1e-14) 

        # Vectorized log calculation
        self.log_likelihoods_ = np.log(probs)
        self.log_neg_likelihoods_ = np.log(1.0 - probs)
        
        self.log_likelihoods_ = pd.DataFrame(self.log_likelihoods_, index=feature_counts.index, columns=feature_counts.columns)
        self.log_neg_likelihoods_ = pd.DataFrame(self.log_neg_likelihoods_, index=feature_counts.index, columns=feature_counts.columns)
        
    def predict(self, X):
        if self.log_likelihoods_ is None:
            raise ValueError("Model has not been trained. Call fit() first.")
            
        posteriors_list = []
        train_cols = self.log_likelihoods_.columns.intersection(X.columns)
        X_aligned = X[train_cols].values 
        
        for c in self.classes_:
            ll = self.log_likelihoods_.loc[c, train_cols].values
            nll = self.log_neg_likelihoods_.loc[c, train_cols].values
            
            # Extract log prior for class c
            log_prior_c = self.log_priors_.loc[c] 
            
            # Vectorized dot product for log-likelihood score
            # log_score = log_prior + (X_aligned * ll).sum(axis=1) + ((1 - X_aligned) * nll).sum(axis=1)
            log_likelihood_score = X_aligned.dot(ll) + (1 - X_aligned).dot(nll)
            
            # Final log-posterior score
            log_score_np = log_likelihood_score + log_prior_c
            
            # Convert back to Series for concatenation
            log_score = pd.Series(log_score_np, index=X.index, name=c)
            posteriors_list.append(log_score)
            
        scores = pd.concat(posteriors_list, axis=1)
        return scores.idxmax(axis=1)

    def save_model(self, filepath, vocab_list, ohe_encoders):
        """
        Saves the trained model weights and preprocessing artifacts to a JSON file.
        """
        if self.log_likelihoods_ is None:
            raise ValueError("Model is not trained yet. Cannot save.")
            
        # Prepare dictionary structure
        # We convert Pandas objects to dicts/lists and use the NumpyEncoder for safe JSON dumping
        model_dump = {
            "alpha": self.alpha,
            "classes": self.classes_, 
            # Convert Series to list/dict
            "log_priors": self.log_priors_.to_dict(), 
            # Convert DataFrames to dictionary (orient='split' preserves index, columns, and data structure)
            "log_likelihoods": self.log_likelihoods_.to_dict(orient='split'),
            "log_neg_likelihoods": self.log_neg_likelihoods_.to_dict(orient='split'),
            # Preprocessing artifacts needed for inference
            "vocab_list": vocab_list,
            "ohe_encoders": ohe_encoders
        }
        
        # Ensure ohe_encoders keys are strings (JSON requirement)
        safe_ohe = {}
        for col, mapping in ohe_encoders.items():
            safe_ohe[col] = {str(k): v for k, v in mapping.items()}
        model_dump["ohe_encoders"] = safe_ohe

        with open(filepath, 'w') as f:
            json.dump(model_dump, f, cls=NumpyEncoder, indent=4)
        
        print(f"Model successfully saved to {filepath}")


if __name__ == "__main__":
    k = 8
    alpha = 1
    trainfile = 'data/train_data.csv' 
    testfile = 'data/test_data.csv'
    
    text_columns = [
        "In your own words, what kinds of tasks would you use this model for?", 
        "Think of one task where this model gave you a suboptimal response. What did the response look like, and why did you find it suboptimal?",
        "When you verify a response from this model, how do you usually go about it?"
    ]
    label_col_name = 'label'

    try:
        df = pd.read_csv(trainfile).drop(columns=['student_id'])
    except FileNotFoundError:
        print(f"Error: The file '{trainfile}' was not found.")
        exit()
        
    if df.empty:
        print("Stopping: No data was loaded.")
        exit()

    split_point = 644
    
    if len(df) > split_point:
        df_train = df.iloc[:split_point].reset_index(drop=True)
        df_val = df.iloc[split_point:].reset_index(drop=True)
                
        X_train, t_train, train_vocab, train_ohe = create_features(
            df_train, text_columns, label_col_name, 
            vocab_list=None, ohe_encoders=None
        )

        X_val, t_val, _, _ = create_features(
            df_val, text_columns, label_col_name, 
            vocab_list=train_vocab, ohe_encoders=train_ohe
        )

        if X_train.empty or X_val.empty:
            print("Skipping manual validation: one of the feature matrices is empty.")
        else:
            model = NaiveBayes(alpha=alpha)
            model.fit(X_train, t_train)

            model_sklearn = BernoulliNB(alpha=alpha)
            model_sklearn.fit(X_train, t_train) 
            
            t_pred = model.predict(X_val)
            accuracy = accuracy_score(t_val, t_pred)
            print(f"Validation: {accuracy * 100:.2f}%")
            
            t_pred_sklearn = model_sklearn.predict(X_val) 
            accuracy_no_k_sklearn = accuracy_score(t_val, t_pred_sklearn)
            print(f"Sklearn validation: {accuracy * 100:.2f}%")

            
    else:
        print(f"Skipping manual validation: Dataset has {len(df)} rows, which is not more than {split_point}.")

     # TESTING TIME :)
    X_train_final, t_train_final, final_vocab, final_ohe_encoders = create_features(
        df, text_columns, label_col_name, 
        vocab_list=None, ohe_encoders=None
    )

    final_model = NaiveBayes(alpha=alpha)
    final_model.fit(X_train_final, t_train_final)

    model_filename = "naive_bayes_weights.json"
    final_model.save_model(model_filename, final_vocab, final_ohe_encoders)

    testfile = 'data/test_data.csv' 

    try:
        df_test = pd.read_csv(testfile)
        # Drop student_id 
        if 'student_id' in df_test.columns:
            df_test = df_test.drop(columns=['student_id'])
            
    except FileNotFoundError:
        print(f"Error: The test file '{testfile}' was not found.")
        print("Skipping test evaluation.")
        exit() 
        
    if df_test.empty:
        print("Stopping: No test data was loaded.")
        exit()

    X_test, t_test, _, _ = create_features(
        df_test, text_columns, label_col_name, 
        vocab_list=final_vocab,
        ohe_encoders=final_ohe_encoders 
    )

    if X_test.empty:
         print("No valid test samples were processed. Stopping.")
         exit()
         
    # Make predictions on the test set
    t_pred_test = final_model.predict(X_test)
    f_score = f1_score(t_test, t_pred_test, average='macro')
    print(f"FINAL MACRO F1 SCORE: {f_score:.4f}") # Added print for F1 score
    cm = confusion_matrix(t_test, t_pred_test)
    class_names = ["ChatGPT", "Gemini", "Claude"]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Naive Bayes Confusion Matrix")
    plt.show()

    if not t_test.empty:
        test_accuracy = accuracy_score(t_test, t_pred_test)
        print(f"\nFINAL TEST ACCURACY: {test_accuracy * 100:.2f}%")
    else:
        print("\nTest data does not have labels. Cannot calculate accuracy.")