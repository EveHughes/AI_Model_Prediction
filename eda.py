
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy

multiple_categorical_col = ['Which types of tasks do you feel this model handles best? (Select all that apply.)', 
                   'For which types of tasks do you feel this model tends to give suboptimal responses? (Select all that apply.)']

single_categorical_col = ['How likely are you to use this model for academic tasks?', 
                          'How often do you expect this model to provide responses with references or supporting evidence?',
                          "How often do you verify this model's responses?",
                          'Based on your experience, how often has this model given you a response that felt suboptimal?']

long_response_col = ['In your own words, what kinds of tasks would you use this model for?',
                     'Think of one task where this model gave you a suboptimal response. What did the response look like, and why did you find it suboptimal?',
                    'When you verify a response from this model, how do you usually go about it?' ]


#reading in data
path = Path(__file__).parent / "data/training_data_clean.csv"
data = pd.read_csv(path)

#number unqique in each column:
def num_unique():
    dic = {}
    for col in data.columns:
        num = data[col].nunique()
        dic[col] = num
    return dic

#number w/ length > n for each long response
def num_long_response(n: int):
    dic = {}
    for col in long_response_col:
        num = (data[col].str.len() > n).sum()
        dic[col] = num

    return dic

#plot single categorical across labels, in progress
def plot_single_category(categories = single_categorical_col):
    x = data['label']
    for category in categories:
        y = data[category]
        y = int(y.split('—')[0].strip())       

        # count responses by label and value
        counts = data.groupby([x, y]).size().unstack(fill_value=0)

        # center at 3
        midpoint = len(y) // 2
        for i, col in enumerate(y):
            offset = i - midpoint
            plt.barh(counts.index, counts[col] * (1 if offset > 0 else -1),
                     left=counts.iloc[:, :i].sum(axis=1) * (1 if offset > 0 else -1),
                     label=col)

        plt.title(category)
        plt.xlabel('Count')
        plt.ylabel('Model')
        plt.legend(title='Response', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
    

    


#run everything
if __name__ == "__main__":
    # print(num_unique())
    # print(num_long_response(10))
    plot_single_category()