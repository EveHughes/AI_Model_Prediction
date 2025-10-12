
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

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
    for category in categories:
        plt.figure(figsize=(14, 7))

        x = data['label']
        y = data[category]

        # convert to int
        int_y = y.apply(lambda v: int(str(v).strip()[0]) if pd.notna(v) and str(v).strip() else np.nan)

        # temp dataframe
        temp = data.copy()
        temp['int_y'] = int_y
        temp = temp.dropna(subset=['int_y'])
        temp['int_y'] = temp['int_y'].astype(int)

        # count responses by label and integer value & convert to %
        counts = temp.groupby(['label', 'int_y']).size().unstack(fill_value=0)
        counts = counts.div(counts.sum(axis=1), axis=0) * 100

        # plot likhert
        midpoint = 3
        
        # separate negative and positive sides
        neg_cols = sorted([col for col in counts.columns if col < midpoint])
        pos_cols = sorted([col for col in counts.columns if col >= midpoint])

        # plot negative side (left) - start from furthest left
        left_pos = -counts[neg_cols].sum(axis=1)
        for col in neg_cols:  # plot in order: 1, 2
            plt.barh(
                counts.index,
                counts[col],
                left=left_pos,
                label=str(col)
            )
            left_pos += counts[col]

        # plot positive side (right) - start from 0 and go right
        left_pos = pd.Series(0, index=counts.index)
        for col in pos_cols:
            plt.barh(
                counts.index,
                counts[col],
                left=left_pos,
                label=str(col)
            )
            left_pos += counts[col]

        plt.title(category)
        plt.xlabel('Percentage (%)')
        plt.ylabel('Model')

        
        responses = sorted([v for v in y.unique() if pd.notna(v)], key=lambda v: int(str(v)[0]))
        plt.legend(title='Response', labels = responses, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xlim(-100, 100)
        plt.xticks(range(-100, 101, 20), [f"{abs(x)}%" for x in range(-100, 101, 20)])

        plt.tight_layout()
        
        filename = "_".join(str(category).split(" ")[:5]) + ".png"
        path = Path(__file__).parent / f"figures/{filename}.png"
        plt.savefig(path, dpi = 300)
        plt.show()

#run everything
if __name__ == "__main__":
    # print(num_unique())
    # print(num_long_response(10))
    plot_single_category()