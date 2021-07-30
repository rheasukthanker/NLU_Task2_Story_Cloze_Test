import numpy as np
import nltk
import pandas as pd
from os.path import join as pathjoin

DATA_PATH = 'data'
TRAIN_PATH = pathjoin(DATA_PATH, 'train_stories.csv')

df = pd.read_csv(TRAIN_PATH)

n_samples = df.values.shape[0]
X_separate = df.values[:, 2:6]
with open(pathjoin(DATA_PATH, 'stories_oneline.txt'), 'w') as f:
    for row in X_separate:
        out = ' '.join(' ' .join(nltk.word_tokenize(sent)) for sent in row)
        f.write(f'{out}\n')
