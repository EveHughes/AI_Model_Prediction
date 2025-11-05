import pandas as pd
import csv
import text_to_vector # Assuming this is needed elsewhere but not for the data loading logic

# The make_bow function remains correct for the updated multi-class labeling
def make_bow(text_data, vocab):
    """
    Produce the bag-of-word representation of the data, along with a vector
    of labels.

    Parameters:
        `data`: a list of `(review, label)` pairs, e.g., [("a good model", "chatgpt"), ...]
        `vocab`: a list consisting of all unique words in the vocabulary

    Returns:
        `X`: A pandas DataFrame of bag-of-word features with shape [len(data), len(vocab)].
             X.iloc[i,j] == 1 if the review in data[i] contains the word vocab[j], 
             and 0 otherwise. The columns are named after the words in `vocab`.
        `t`: A pandas Series of shape [len(data)], with t.iloc[i] == 1 if 
             data[i] is a positive review, and 0 otherwise.
    """
    vocab_map = {word: j for j, word in enumerate(vocab)}
    X_list = []
    t_list = []
    
    for response, label in text_data:
        bow_vector = [0] * len(vocab)
        words = response.split(' ') 
        unique_words = set(word.strip().lower() for word in words) # Cleans words and ensures binary features
        
        for word in unique_words:
            if word in vocab_map:
                j = vocab_map[word]
                bow_vector[j] = 1
        X_list.append(bow_vector)
        
        label_lower = label.lower()
        if label_lower == 'chatgpt': 
            t_list.append(0)
        elif label_lower == 'claude':
            t_list.append(1)
        elif label_lower == 'gemini':
            t_list.append(2)

    X = pd.DataFrame(X_list, columns=vocab)
    t = pd.Series(t_list)
    
    return X, t 

if __name__ == "__main__":
    trainfile = 'data/train_data.csv'
    testfile = 'data/test_data.csv'

    with open(trainfile, 'r', encoding='utf-8') as f:
        data_reader = csv.reader(f)
        
        try:
            header_row = next(data_reader)
        except StopIteration:
            print("Error: The CSV file is empty.")
            exit()
        
        text_columns = [
            "In your own words, what kinds of tasks would you use this model for?", 
            "Think of one task where this model gave you a suboptimal response. What did the response look like, and why did you find it suboptimal?",
            "When you verify a response from this model, how do you usually go about it?"
        ]
        label_col_name = 'label'

        text_column_indices = [header_row.index(col_name) for col_name in text_columns]
        label_index = header_row.index(label_col_name)

        unique_vocab = set()
        
        col1_pairs = [] 
        col2_pairs = [] 
        col3_pairs = []
        all_pairs = [col1_pairs, col2_pairs, col3_pairs] 

        f.seek(0)
        data_reader = csv.reader(f)
        next(data_reader) # Skip header

        for row in data_reader:
            label = row[label_index].strip()
            
            for i, text_col_idx in enumerate(text_column_indices):
                text = row[text_col_idx].strip()
                
                words = text.split(' ')
                for word in words:
                    unique_vocab.add(word.strip().lower())
                
                all_pairs[i].append((text, label))
        print(col1_pairs[:2])
        print(col2_pairs[:2])
        print(col3_pairs[:2])
