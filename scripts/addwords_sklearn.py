from pathlib import Path
import pandas as pd
import numpy as np

from itertools import product

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score


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
TEST_PATH = Path(__file__).parent.parent / "data/test_data.csv"
TEST_SAVE = Path(__file__).parent / "test_data_cleaned.csv"
TRAIN_PATH =  BASE_PATH / "mlp_cleaned_data/add_words_train_data.csv"
GRID_PATH = BASE_PATH / "mlp_grid_search/add_words_grid.csv"
WEIGHT_BIAS_PATH = BASE_PATH / "mlp_add_words_vals/model_weights_biases.npz"
MEAN_STD_PATH = BASE_PATH / "mlp_add_words_vals/data_mean_std.csv"

# Grid Search
one_layer_sizes = [(i,) for i in [20, 30, 40, 50, 60]]
two_layer_sizes = list(product([20, 30, 40, 50, 60], [5, 10, 20]))
all_hidden_sizes = one_layer_sizes + two_layer_sizes
GRID_PARAMS = {
    "hidden_layer_sizes": all_hidden_sizes,
    "learning_rate_init": [0.001, 0.01],
    "alpha": [1e-4, 1e-5],
    "activation": ["relu"],
    "solver": ["adam"]
}


##################################################
########### SAVING DATA AND FILLING ##############
##################################################
#feature engineering with word responses
def encode_words(data: pd.DataFrame) -> pd.DataFrame:
    # Get the three text columns before they're dropped
    data['all_text'] = (data[NON_NUMERIC[0]].fillna('') + ' ' + 
                        data[NON_NUMERIC[1]].fillna('') + ' ' + 
                        data[NON_NUMERIC[2]].fillna('')).str.lower()
    
    # CATEGORY 1: RESPONSE QUALITY
    quality_words = {
        'verbose': ['verbose', 'too long', 'wordy', 'lengthy'],
        'generic': ['generic', 'vague', 'general', 'basic'],
        'wrong': ['wrong', 'incorrect', 'mistake', 'error'],
        'incomplete': ['incomplete', 'partial', 'missing', 'omit'],
        'complex': ['complex', 'complicated', 'convoluted', 'overcomplicate'],
        'repetitive': ['repeat', 'repetitive', 'redundant', 'duplicate'],
        'robotic': ['robotic', 'unnatural', 'obviously ai'],
        'confusing': ['confusing', 'unclear', 'ambiguous', 'hard to understand'],
        'superficial': ['superficial', 'shallow', 'surface level', 'lack depth']
    }
    
    for feature, keywords in quality_words.items():
        data[f'quality_{feature}'] = data['all_text'].apply(
            lambda x: 1 if any(word in x for word in keywords) else 0
        )
    
    # CATEGORY 2: TECHNICAL ISSUES
    technical_words = {
        'bug': ['bug', 'buggy', 'broken'],
        'syntax': ['syntax', 'compile', 'compilation error'],
        'crash': ['crash', 'fail', 'failure', 'doesn\'t work'],
        'slow': ['slow', 'inefficient', 'performance'],
        'memory': ['memory', 'forget', 'lost context'],
        'hallucination': ['hallucin', 'made up', 'fabricat', 'invented'],
        'outdated': ['outdated', 'old', 'deprecated'],
        'formatting': ['format', 'latex', 'markdown', 'typeset']
    }
    
    for feature, keywords in technical_words.items():
        data[f'technical_{feature}'] = data['all_text'].apply(
            lambda x: 1 if any(word in x for word in keywords) else 0
        )
    
    # CATEGORY 3: TASK DOMAINS
    task_words = {
        'coding': ['code', 'coding', 'program', 'script', 'debug'],
        'math': ['math', 'computation', 'calculate', 'equation', 'proof'],
        'writing': ['write', 'writing', 'essay', 'draft', 'compose'],
        'research': ['research', 'search', 'find information', 'investigate'],
        'analysis': ['analysis', 'analyze', 'data processing'],
        'creative': ['creative', 'brainstorm', 'idea', 'imagination'],
        'learning': ['learn', 'understand', 'explain', 'concept'],
        'professional': ['professional', 'email', 'resume', 'cover letter'],
        'documentation': ['documentation', 'summarize', 'summary']
    }
    
    for feature, keywords in task_words.items():
        data[f'task_{feature}'] = data['all_text'].apply(
            lambda x: 1 if any(word in x for word in keywords) else 0
        )
    
    # CATEGORY 4: VERIFICATION METHODS
    verify_words = {
        'google': ['google', 'search engine', 'online search'],
        'test_code': ['run', 'test', 'execute', 'try the code'],
        'compare_ai': ['another model', 'other ai', 'compare', 'cross-check'],
        'manual': ['manual', 'by hand', 'myself', 'manually'],
        'sources': ['source', 'reference', 'citation', 'link'],
        'forums': ['forum', 'stackoverflow', 'stack overflow', 'community'],
        'documentation': ['documentation', 'docs', 'official'],
        'ask_human': ['ask friend', 'ask someone', 'human', 'expert']
    }
    
    for feature, keywords in verify_words.items():
        data[f'verify_{feature}'] = data['all_text'].apply(
            lambda x: 1 if any(word in x for word in keywords) else 0
        )
    
    # CATEGORY 5: SPECIFIC TECHNOLOGIES
    tech_words = {
        'python': ['python'],
        'javascript': ['javascript', 'js', 'react', 'node'],
        'latex': ['latex'],
        'html_css': ['html', 'css', 'frontend', 'web dev'],
        'sql': ['sql', 'database'],
        'api': ['api'],
        'machine_learning': ['machine learning', 'ml', 'ai model'],
        'excel': ['excel', 'spreadsheet']
    }
    
    for feature, keywords in tech_words.items():
        data[f'tech_{feature}'] = data['all_text'].apply(
            lambda x: 1 if any(word in x for word in keywords) else 0
        )
    
    # CATEGORY 6: USER SENTIMENT
    sentiment_words = {
        'positive': ['good', 'great', 'best', 'fantastic', 'excellent', 'helpful'],
        'negative': ['bad', 'worst', 'terrible', 'useless', 'frustrating'],
        'neutral': ['okay', 'fine', 'acceptable', 'decent'],
        'disappointed': ['disappoint', 'suboptimal', 'not good enough'],
        'satisfied': ['satisfied', 'works well', 'reliable', 'trust']
    }
    
    for feature, keywords in sentiment_words.items():
        data[f'sentiment_{feature}'] = data['all_text'].apply(
            lambda x: 1 if any(word in x for word in keywords) else 0
        )
    
    # CATEGORY 7: USAGE FREQUENCY
    frequency_words = {
        'always': ['always', 'every time', 'constantly'],
        'often': ['often', 'frequently', 'regularly'],
        'sometimes': ['sometimes', 'occasionally'],
        'rarely': ['rarely', 'seldom', 'not often'],
        'never': ['never', 'don\'t use', 'haven\'t used']
    }
    
    for feature, keywords in frequency_words.items():
        data[f'frequency_{feature}'] = data['all_text'].apply(
            lambda x: 1 if any(word in x for word in keywords) else 0
        )
    
    # CATEGORY 8: SPECIFIC PROBLEMS
    problem_words = {
        'too_simple': ['too simple', 'too basic', 'oversimplif'],
        'too_advanced': ['too advanced', 'too complex', 'beyond'],
        'not_follow_instructions': ['doesn\'t follow', 'ignored', 'not listen'],
        'context_lost': ['lost context', 'forgot', 'doesn\'t remember'],
        'cant_debug': ['can\'t debug', 'unable to fix', 'couldn\'t solve'],
        'over_engineered': ['over-engineer', 'overengine', 'unnecessarily complex'],
        'wrong_approach': ['wrong approach', 'wrong method', 'inefficient method']
    }
    
    for feature, keywords in problem_words.items():
        data[f'problem_{feature}'] = data['all_text'].apply(
            lambda x: 1 if any(word in x for word in keywords) else 0
        )
    
    # Drop the temporary all_text column
    data = data.drop('all_text', axis=1)
    
    return data


# Save data w/ dropped null columns and encoded categories
def prepare_data():
    data = pd.read_csv(DATA_PATH)
    data = encode_words(data)
    data = data.drop(NON_NUMERIC + NA_COLS, axis = 1)
    data.to_csv(TRAIN_PATH)

# Save data w/ dropped null columns and encoded categories
def prepare_test():
    data = pd.read_csv(TEST_PATH)
    data = encode_words(data)
    data = data.drop(NON_NUMERIC + NA_COLS, axis = 1)
    data.to_csv(TEST_SAVE)

# saves mean and standard dev for scaling w/ dropped na
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
########## CREATING MODEL WITH NO NULL ###########
##################################################

# Model that DOESN'T take in null columns
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
        cv=6,
        scoring="accuracy",
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

    return grid.best_estimator_


##################################################
######### TRAINING BEST FROM GRIDSEARCH ###########
##################################################

def best_model(model):
    scaler = StandardScaler()
    #get train data
    train_data = pd.read_csv(TRAIN_PATH, index_col = 0)
    t_train = train_data["label"]
    X_train = train_data.drop(LABELS, axis = 1)
    X_train_scaled = scaler.fit_transform(X_train)
    
    #model accuracy on train data
    train_accuracy = model.score(X_train_scaled, t_train)
    print(f"Training Accuracy: {train_accuracy:.4f}")

    #get test data
    test_data = pd.read_csv(TEST_SAVE, index_col = 0)
    t_test = test_data["label"]
    X_test = test_data.drop(LABELS, axis = 1)
    X_test_scaled = scaler.fit_transform(X_test)

    #model accuracy on test data
    test_accuracy = model.score(X_test_scaled, t_test)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    #f1 score on test data
    y_pred = model.predict(X_test_scaled)
    macro_f1 = f1_score(t_test, y_pred, average='macro')
    print("Macro F1:", macro_f1)

    # Confusion matrix of test data
    import matplotlib.pyplot as plt
    cm = confusion_matrix(t_test, y_pred)
    print("Confusion Matrix:\n", cm)
    disp = ConfusionMatrixDisplay(cm, display_labels=model.classes_)
    disp.plot(cmap=plt.cm.Blues)  
    plt.title("Multilayer Perceptron Confusion Matrix")
    plt.show()


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

if __name__ == "__main__":
    prepare_data()
    prepare_test()
    # get_mean_std()
    model = gridsearch()
    best_model(model)
