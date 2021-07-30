import numpy as np
import multiprocessing
import pandas as pd
from os.path import join as pathjoin
import os
import subprocess
import nltk
import csv

"""
Script to combine train_stories.csv which contains only positive stories
with train_neg_stories.csv which contains only negative stories into the
validation set format.
"""


DATA_PATH = 'data'
TRAIN_PATH = pathjoin(DATA_PATH, 'train_stories.csv')
TRAIN_NEG_PATH = pathjoin(DATA_PATH, 'train_stories_neg_nearest_story_usc_with_nlp_features.csv')
COMBINED_PATH = pathjoin(DATA_PATH, 'train_stories_nearest_story_usc_with_nlp_features_combined.csv')


def combine(row_pos, row_neg):
    return [row_pos[0], row_pos[2], row_pos[3], row_pos[4], row_pos[5], row_pos[6], row_neg[6], 1]


def main():
    df_pos = pd.read_csv(TRAIN_PATH)
    df_neg = pd.read_csv(TRAIN_NEG_PATH)
    X_out = []
    with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
        X_out = p.starmap(combine, zip(df_pos.values, df_neg.values))
    X_out = np.array(X_out)
    df_out = pd.DataFrame(
            index=df_pos.index,
            columns=['InputStoryid', 'InputSentence1', 'InputSentence2', 'InputSentence3', 'InputSentence4', 'RandomFifthSentenceQuiz1', 'RandomFifthSentenceQuiz2', 'AnswerRightEnding'],
            data=X_out
    )
    df_out.to_csv(COMBINED_PATH, index=False)


if __name__ == '__main__':
    main()
