import pandas as pd
import os # standard python library
import re # standard python library
import math # standard python library
from collections import Counter # standard python library
import numpy as np

CSV_FILE_PATH = 'data/cleaned_data.csv'
TEXT_COLUMN = ["In your own words, what kinds of tasks would you use this model for?", 
               "Think of one task where this model gave you a suboptimal response. What did the response look like, and why did you find it suboptimal?",
               "When you verify a response from this model, how do you usually go about it?"]
LABEL_COLUMN = 'label'

def tokenize(text):
    """
    Tokenize the text by making everything lower-case, remove punctuation, and split each response into individual words.
    """
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    words = text.split()
    return words

def process_data_manual(filepath, text_col_list, label_col):
    """
    Loads data, vectorizes text, and encodes labels manually using numpy.
    It now accepts a list of text columns (`text_col_list`) and concatenates
    the text from these columns for each row to form a single document.
    """
    try:
        df = pd.read_csv(filepath)
        df[text_col_list] = df[text_col_list].fillna('')

        # axis=1 -> lambda is applied row-wise and row.values contains the string data
        combined_text_series = df[text_col_list].apply(
            lambda row: ' '.join(row.values.astype(str)),
            axis=1
        )

        # Tokenize the combined text for each row
        tokenized_docs = [tokenize(doc) for doc in combined_text_series]
        num_docs = len(tokenized_docs)

        unique_labels = sorted(df[label_col].unique())
        label_mapping = {label: i for i, label in enumerate(unique_labels)}

        y = np.array([label_mapping[label] for label in df[label_col]])
        print(f"Label mapping: {label_mapping}")

        # create a set of all unique words and mapping
        vocab_set = set(word for doc in tokenized_docs for word in doc)
        vocab_list = sorted(list(vocab_set))

        # map word -> index
        vocab = {word: i for i, word in enumerate(vocab_list)}
        # map index -> word
        inv_vocab = {i: word for word, i in vocab.items()}
        vocab_size = len(vocab)
        print(f"Built vocabulary with {vocab_size} unique words.")

        tf_matrix = np.zeros((num_docs, vocab_size))

        for doc_index, doc in enumerate(tokenized_docs):
            doc_word_counts = Counter(doc)
            total_words_in_doc = len(doc)

            if total_words_in_doc > 0:
                for word, count in doc_word_counts.items():
                    if word in vocab:
                        word_index = vocab[word]
                        tf_matrix[doc_index, word_index] = count / total_words_in_doc

        doc_freq = np.zeros(vocab_size)
        for word, index in vocab.items():
            for doc in tokenized_docs:
                if word in doc:
                    doc_freq[index] += 1

        idf_vector = np.log(num_docs / (1 + doc_freq)) + 1

        tfidf_matrix = tf_matrix * idf_vector

        print(f"Feature matrix (X) shape: {tfidf_matrix.shape}")
        print(f"Target vector (y) shape: {y.shape}")

        return tfidf_matrix, y, vocab, label_mapping

    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return None
    except KeyError as e:
        print(f"Error: Column not found. Make sure your CSV has columns in '{text_col_list}' and '{label_col}'.")
        print(f"Details: {e}")
        return None

def main():
    """
    Main function to run the setup and processing.
    """
    result = process_data_manual(CSV_FILE_PATH, TEXT_COLUMN, LABEL_COLUMN)

    if result:
        tfidf_matrix, y, vocab, label_mapping = result
        print("The 'tfidf_matrix' (a numpy array) and 'y' (a numpy array) are ready.")

if __name__ == "__main__":
    main()