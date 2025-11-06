from pathlib import Path
import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

##################################################
################# VARIABLES  ####################
##################################################

# Columns
NON_NUMERIC = [ 'In your own words, what kinds of tasks would you use this model for?',
                     'Think of one task where this model gave you a suboptimal response. What did the response look like, and why did you find it suboptimal?',
                    'When you verify a response from this model, how do you usually go about it?']

NA_COLS = ['Based on your experience, how often has this model given you a response that felt suboptimal?', 'How often do you expect this model to provide responses with references or supporting evidence?', "How often do you verify this model's responses?"]

LABELS = ['student_id', 'label']

# File Paths
DATA_PATH = Path(__file__).parent.parent / "data/train_data.csv"
TRAIN_PATH = Path(__file__).parent / "mlp_data/mlp_train_data.csv"
FILLED_TRAIN_PATH = Path(__file__).parent / "mlp_data/mlp_filled_train_data.csv"
MODEL1_GRID_PATH = Path(__file__).parent / "mlp_data/model1_grid.csv"
MODEL2_GRID_PATH = Path(__file__).parent / "mlp_data/model2_grid.csv"


##################################################
######### SAVING DATA AND FILLING #############
##################################################

# Save unfilled data
def save_data():
    data = pd.read_csv(DATA_PATH)
    data = data.drop(NON_NUMERIC, axis = 1)
    data.to_csv(TRAIN_PATH)

# Get columns with null values
def get_null_columns():
    data = pd.read_csv(DATA_PATH)
    data = data.drop(NON_NUMERIC, axis = 1)
    cols = data.columns[data.isna().any()].tolist()
    
    print(cols)

# Save filled data
def save_filled_data():
    data = pd.read_csv(DATA_PATH)
    data = data.drop(NON_NUMERIC, axis = 1)
    
    # replace each with average value
    for column in NA_COLS:
        val = data[column].mean()
        data[column] = data[column].fillna(val)

    data.to_csv(FILLED_TRAIN_PATH)


##################################################
########## CREATING MODEL WITH NO NULL ###########
##################################################

#Model that DOESN'T take in null columns
def create_model1():
    #get data
    train_data = pd.read_csv(TRAIN_PATH, index_col = 0)
    t_train = train_data["label"]

    scaler = StandardScaler()
    X_train = train_data.drop(LABELS + NA_COLS, axis = 1)
    X_train_scaled = scaler.fit_transform(X_train)

    #grid search
    param_grid = {
        "hidden_layer_sizes": [(50, 20), (60, 20), (70, 20)],
        "learning_rate_init": [0.001, 0.01],
        "alpha" : [1e-3, 1e-4, 1e-5],
        "solver" : ['adam', 'sgd']
    }
    model = MLPClassifier(max_iter=1000, random_state=1, batch_size = 33)
    grid = GridSearchCV(
        model,
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1
    )
    grid.fit(X_train_scaled, t_train)

    #best parameters
    print("Best params:", grid.best_params_)
    print("Best CV score:", grid.best_score_)
    results = pd.DataFrame(grid.cv_results_)
    results = results[['param_hidden_layer_sizes', 'param_learning_rate_init', 'param_solver', 'param_alpha', 'mean_test_score', 'std_test_score']]
    results = results.sort_values(by='mean_test_score', ascending=False)
    results.to_csv(MODEL1_GRID_PATH)

    #model
    model1 = grid.best_estimator_


##################################################
######### CREATING MODEL THAT FILLS NULL ##########
##################################################

def create_model2():
    #get data
    train_data = pd.read_csv(FILLED_TRAIN_PATH, index_col = 0)
    t_train = train_data["label"]

    scaler = StandardScaler()
    X_train = train_data.drop(LABELS, axis = 1)
    X_train_scaled = scaler.fit_transform(X_train)

    #grid search
    param_grid = {
        "hidden_layer_sizes": [(50, 20), (60, 20), (70, 20)],
        "learning_rate_init": [0.001, 0.01],
        "alpha" : [1e-3, 1e-4, 1e-5],
        "solver" : ['adam', 'sgd']
    }
    model = MLPClassifier(max_iter=1000, random_state=1, batch_size = 33)
    grid = GridSearchCV(
        model,
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1
    )
    grid.fit(X_train_scaled, t_train)

    #best parameters
    print("Best params:", grid.best_params_)
    print("Best CV score:", grid.best_score_)
    results = pd.DataFrame(grid.cv_results_)
    results = results[['param_hidden_layer_sizes', 'param_learning_rate_init', 'param_solver', 'param_alpha', 'mean_test_score', 'std_test_score']]
    results = results.sort_values(by='mean_test_score', ascending=False)
    results.to_csv(MODEL2_GRID_PATH)

    #model
    model2 = grid.best_estimator_
    

if __name__ == "__main__":
    # save_data()
    # save_filled_data()
    create_model1()
    create_model2()
