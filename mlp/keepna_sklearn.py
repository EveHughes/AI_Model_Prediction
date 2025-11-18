from pathlib import Path
import pandas as pd
import numpy as np

from itertools import product

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

##################################################
################# VARIABLES  ####################
##################################################

# Columns
NON_NUMERIC = [ 'In your own words, what kinds of tasks would you use this model for?',
                     'Think of one task where this model gave you a suboptimal response. What did the response look like, and why did you find it suboptimal?',
                    'When you verify a response from this model, how do you usually go about it?']

NA_COLS = ['Based on your experience, how often has this model given you a response that felt suboptimal?', 
           'How often do you expect this model to provide responses with references or supporting evidence?', 
           "How often do you verify this model's responses?"]

LABELS = ['student_id', 'label']

# File Paths
BASE_PATH = Path(__file__).parent
DATA_PATH = Path(__file__).parent.parent / "data/train_data.csv"
TRAIN_PATH =  BASE_PATH / "mlp_cleaned_data/keepna_train_data.csv"
GRID_PATH = BASE_PATH / "mlp_grid_search/keepna_grid.csv"
WEIGHT_BIAS_PATH = BASE_PATH / "mlp_keepna_vals/model_weights_biases.npz"
MEAN_STD_PATH = BASE_PATH / "mlp_keepna_vals/data_mean_std.csv"

# Grid Search
all_tuples = list(product(range(55, 60), range(15, 20)))
GRID_PARAMS = {
    "hidden_layer_sizes": all_tuples,
    "learning_rate_init": [0.001, 0.01],
    "alpha" : [1e-5, 1e-4],
    "activation": ["relu", "tanh"],
    "solver" : ['adam', 'sgd']
    }


##################################################
########### SAVING DATA AND FILLING ##############
##################################################

# Get columns with null values
def get_null_columns():
    data = pd.read_csv(DATA_PATH)
    data = data.drop(NON_NUMERIC, axis = 1)
    cols = data.columns[data.isna().any()].tolist()
    
    print(cols)

# Save filled data
def prepare_data():
    data = pd.read_csv(DATA_PATH)
    data = data.drop(NON_NUMERIC, axis = 1)
    
    # replace each with average value
    for column in NA_COLS:
        val = data[column].mean()
        data[column] = data[column].fillna(val)

    data.to_csv(TRAIN_PATH)

# saves mean and standard dev for scaling
def get_mean_std():
    train_data = pd.read_csv(TRAIN_PATH, index_col = 0)
    X_train = train_data.drop(LABELS, axis = 1)
    
    scaler = StandardScaler()
    scaler.fit(X_train)
    scaling_params = pd.DataFrame({
    'mean': scaler.mean_,
    'std': scaler.scale_
    }, index=X_train.columns)

    scaling_params.to_csv(MEAN_STD_PATH)

##################################################
######### CREATING MODEL THAT FILLS NULL #########
##################################################

def gridsearch():
    #get data
    train_data = pd.read_csv(TRAIN_PATH, index_col = 0)
    t_train = train_data["label"]

    scaler = StandardScaler()
    X_train = train_data.drop(LABELS, axis = 1)
    X_train_scaled = scaler.fit_transform(X_train)

    #grid search
    model = MLPClassifier(max_iter=1000, random_state=1, batch_size = 33)
    grid = GridSearchCV(
        model,
        GRID_PARAMS,
        cv=5,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1
    )
    grid.fit(X_train_scaled, t_train)

    #best parameters
    print("Best params:", grid.best_params_)
    print("Best CV score:", grid.best_score_)
    results = pd.DataFrame(grid.cv_results_)
    results = results[['param_hidden_layer_sizes', 'param_activation', 'param_learning_rate_init', 'param_solver', 'param_alpha', 'mean_test_score', 'std_test_score']]
    results = results.sort_values(by='mean_test_score', ascending=False)
    results.to_csv(GRID_PATH)


##################################################
###### TRAINING BEST MODEL FROM GRIDSEARCH #######
##################################################

def best_model():
    #get data
    train_data = pd.read_csv(TRAIN_PATH, index_col = 0)
    t_train = train_data["label"]

    scaler = StandardScaler()
    X_train = train_data.drop(LABELS, axis = 1)
    X_train_scaled = scaler.fit_transform(X_train)

    model = MLPClassifier(max_iter=1000, 
                          random_state=1, 
                          batch_size = 33, 
                          hidden_layer_sizes=(59, 18),
                          learning_rate_init=0.001,
                          solver='sgd',
                          alpha=1e-05
    )
    model.fit(X_train_scaled, t_train)

    #model accuracy on training data
    train_accuracy = model.score(X_train_scaled, t_train)
    print(f"Training Accuracy: {train_accuracy:.4f}")

    
    # Confusion matrix
    y_pred = model.predict(X_train_scaled)
    cm = confusion_matrix(t_train, y_pred)
    print("Confusion Matrix:\n", cm)

    #exporting weights and biases
    np.savez_compressed(
        WEIGHT_BIAS_PATH,
        weights_0=model.coefs_[0],
        weights_1=model.coefs_[1],
        weights_2=model.coefs_[2],
        biases_0=model.intercepts_[0],
        biases_1=model.intercepts_[1],
        biases_2=model.intercepts_[2]
    )

    # index of classes
    # print(model.classes_)


if __name__ == "__main__":
    prepare_data()
    get_mean_std()
    gridsearch()
    best_model()