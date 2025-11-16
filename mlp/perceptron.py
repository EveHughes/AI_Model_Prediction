import numpy as np
import pandas as pd
from pathlib import Path


##################################################
################### CONSTANTS ####################
##################################################

# variables
NON_NUMERIC = [ 'In your own words, what kinds of tasks would you use this model for?',
                     'Think of one task where this model gave you a suboptimal response. What did the response look like, and why did you find it suboptimal?',
                    'When you verify a response from this model, how do you usually go about it?']

NA_COLS = ['Based on your experience, how often has this model given you a response that felt suboptimal?', 
           'How often do you expect this model to provide responses with references or supporting evidence?', 
           "How often do you verify this model's responses?"]

LABELS = ['student_id', 'label']

MEAN_VALUES = {"Based on your experience, how often has this model given you a response that felt suboptimal?": 2.8627187079407808,
               "How often do you expect this model to provide responses with references or supporting evidence?": 2.8055555555555554,
               "How often do you verify this model's responses?": 3.174324324324324}

CLASSES = ['ChatGPT', 'Claude', 'Gemini']

# paths
BASE_PATH = Path(__file__).parent
TRAIN_DATA_PATH = Path(__file__).parent.parent / "data/train_data.csv"
TEST_DATA_PATH = Path(__file__).parent.parent / "data/test_data.csv"
MODEL_WEIGHT_BIAS_PATH = BASE_PATH / "mlp_model_vals/model_weights_biases.npz"
DATA_MEAN_STD_PATH = BASE_PATH / "mlp_model_vals/data_mean_std.csv"



##################################################
################ PREPARING VALUES ################
##################################################

# fill null values with mean values -- found mean from training data
def fill_null(df: pd.DataFrame):
    # replace each with average value
    for column in NA_COLS:
        val = MEAN_VALUES[column]
        df[column] = df[column].fillna(val)
    return df

# implementation of sklearn StandardScaler.fit_transform() -- values from training data
def scale(df):
    scaling_params = pd.read_csv(DATA_MEAN_STD_PATH, index_col=0)
    df = df[scaling_params.index]
    df_scaled = (df - scaling_params['mean']) / scaling_params['std']
    return df_scaled

def get_data(filename):
    df = pd.read_csv(filename)
    df = df.drop(NON_NUMERIC + LABELS, axis = 1)
    df = fill_null(df)
    df = scale(df)
    return df
   
##################################################
############### MAKING PREDICTIONS ###############
##################################################

# relu function - between hidden layers
def relu(x):
    return np.maximum(0, x)

# softmax function - output layer
def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

# predicting from input - single prediction
def predict(X_input):
    # getting values
    values = np.load(MODEL_WEIGHT_BIAS_PATH)
    weights_0 = values['weights_0']
    weights_1 = values['weights_1']
    weights_2 = values['weights_2']
    biases_0 =  values['biases_0']
    biases_1 =  values['biases_1']
    biases_2 =  values['biases_2']

    # Layer 1: input -> hidden layer 1 (59 neurons)
    z1 = np.dot(X_input, weights_0) + biases_0
    a1 = relu(z1)  
    
    # Layer 2: hidden layer 1 -> hidden layer 2 (18 neurons)
    z2 = np.dot(a1, weights_1) + biases_1
    a2 = relu(z2)
    
    # Layer 3: hidden layer 2 -> output
    z3 = np.dot(a2, weights_2) + biases_2
    a3 = softmax(z3)
    
    # make prediction
    prediction = CLASSES[np.argmax(a3)]
    return prediction

# Prediction for all X_inputs
def predict_all(filename):
    df = get_data(filename)

    
    predictions = []
    for idx, row in df.iterrows():
        pred = predict(row)
        predictions.append(pred)
    
    return predictions


##################################################
############# EVALUATING PREDICTIONS ##############
##################################################

#returns accuracy of predictions
def accuracy(filepath):
    data = pd.read_csv(filepath, index_col = 0)
    t_data = data["label"]
    predictions = predict_all(filepath)

    acc = np.mean(predictions == t_data.values)
    print(f"Accuracy: {acc:.4f}")
    return acc
    
if __name__ == "__main__":
    accuracy(TRAIN_DATA_PATH)