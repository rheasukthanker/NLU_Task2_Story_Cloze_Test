import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import multiprocessing
import pandas as pd
from os.path import join as pathjoin
import os
import subprocess
import nltk
import csv


DATA_PATH = 'data'
TRAIN_PATH = pathjoin(DATA_PATH, 'train_stories.csv')
TRAIN_NEG_PATH = pathjoin(DATA_PATH, 'train_stories_neg.csv')


class AugBackward:
    """
    Augment a story using Backward strategy.
    """
    def __init__(self):
        pass

    def aug(self, i, row):
        sentences = row[2:7]
        rand_idx = np.random.randint(4)
        sentences[-1] = sentences[rand_idx]
        row[2:7] = sentences
        return row


class AugRand:
    """
    Augment a story using Random strategy.
    """
    def __init__(self, sent_arr):
        self.sent_arr = sent_arr

    def aug(self, i, row):
        n_sentences = len(self.sent_arr)
        rand_idx = np.random.randint(n_sentences)
        row[6] = self.sent_arr[rand_idx][6]
        return row


class AugNearestLastSkipthought:
    """
    Augment a story by picking the ending that is closest to the ending
    of the current story. Closeness is measured in SkipThought vector space
    by cosine similarity.
    """
    def __init__(self, sent_arr):
        self.sent_arr = sent_arr
        n_samples = len(sent_arr)

        last_sent_path = pathjoin(DATA_PATH, 'last_sent_tmp.txt')
        if not os.path.exists(last_sent_path):
            with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
                tokenized_flattened = p.map(self.tokflat_row, sent_arr)
            with open(last_sent_path, 'w') as f:
                for elem in tokenized_flattened:
                    f.write('{}\n'.format(elem))

        closest_path = pathjoin(DATA_PATH, 'sent2vec_closest_last.txt')
        if not os.path.exists(closest_path):
            ps = subprocess.Popen(['cat', last_sent_path], stdout=subprocess.PIPE)
            with open(closest_path, 'w') as f:
                subprocess.run(['sent2vec/fasttext', 'nnSent', 'data/torontobooks_unigrams.bin', last_sent_path, '2'], stdin=ps.stdout, stdout=f)

        self.closest = []
        with open(closest_path, 'r') as f:
            f.readline()
            for _ in range(n_samples):
                f.readline()
                line = f.readline()
                f.readline()
                sent_id = int(line.strip().split()[1])
                self.closest.append(sent_id)
        self.closest = np.array(self.closest)


    def tokflat_row(self, row):
        return ' '.join(nltk.word_tokenize(row[6]))

    def aug(self, i, row):
        row[6] = self.sent_arr[self.closest[i]][6]
        return row


class AugNearestStorySkipthought:
    """
    Augment a story by picking the ending of the story that is closest
    to the current story. Closeness is measured in SkipThought vector space
    by cosine similarity.
    """
    def __init__(self, sent_arr):
        self.sent_arr = sent_arr
        n_samples = len(sent_arr)

        story_path = pathjoin(DATA_PATH, 'story_tmp.txt')
        if not os.path.exists(story_path):
            with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
                tokenized_flattened = p.map(self.tokflat_row, sent_arr)
            with open(story_path, 'w') as f:
                for elem in tokenized_flattened:
                    f.write('{}\n'.format(elem))

        closest_path = pathjoin(DATA_PATH, 'sent2vec_closest_story.txt')
        if not os.path.exists(closest_path):
            ps = subprocess.Popen(['cat', story_path], stdout=subprocess.PIPE)
            with open(closest_path, 'w') as f:
                subprocess.run(['sent2vec/fasttext', 'nnSent', 'data/torontobooks_unigrams.bin', story_path, '2'], stdin=ps.stdout, stdout=f)

        self.closest = []
        with open(closest_path, 'r') as f:
            f.readline()
            for _ in range(n_samples):
                f.readline()
                line = f.readline()
                f.readline()
                sent_id = int(line.strip().split()[1])
                self.closest.append(sent_id)
        self.closest = np.array(self.closest)

    def tokflat_row(self, row):
        return ' '.join(' '.join(nltk.word_tokenize(sent)) for sent in row[2:6])

    def aug(self, i, row):
        row[6] = self.sent_arr[self.closest[i]][6]
        return row


class AugNearestStoryUSCEncoding:
    """
    Augment a story by picking the ending of the story that is closest
    to the current story. Closeness is measured in USC vector space
    by cosine similarity.
    """
    def __init__(self, sent_arr):
        self.sent_arr = sent_arr
        usc_embedding_path = pathjoin(DATA_PATH, 'usc_embeddings.txt')
        if not os.path.exists(usc_embedding_path):
            module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"
            embed = hub.Module(module_url)

            stories = [' '.join(row[2:6]) for row in sent_arr]
            with tf.Session() as session:
                session.run([tf.global_variables_initializer(), tf.tables_initializer()])
                story_embeddings = session.run(embed(stories))

            df = pd.DataFrame(data=story_embeddings)
            df.to_csv(usc_embedding_path, sep=' ', header=None, index=False)

        self.story_embeddings = pd.read_csv(usc_embedding_path, sep=' ', header=None).values

        print('Computing closest')
        self.n_samples = len(sent_arr)
        with multiprocessing.Pool(4) as p:
            self.closest = p.map(self._closest_for_i, range(self.n_samples))
        self.closest = np.array(self.closest)

    def _closest_for_i(self, i):
        sim = np.full(self.n_samples, -np.inf)
        closest = -1
        for j in range(self.n_samples):
            if i == j:
                continue
            dot = np.dot(self.story_embeddings[i], self.story_embeddings[j])
            if dot > sim[i]:
                sim[i] = dot
                closest = j
        return closest

    def aug(self, i, row):
        row[6] = self.sent_arr[self.closest[i]][6]
        return row


class AugNearestStoryUSCWithNLPEncoding:
    """
    Augment a story by picking the ending of the story that is closest
    to the current story. Closeness is measured in USC + NLP vector space
    by cosine similarity.
    """
    def __init__(self, sent_arr):
        self.sent_arr = sent_arr
        usc_embedding_path = pathjoin(DATA_PATH, 'usc_embeddings.txt')
        if not os.path.exists(usc_embedding_path):
            module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"
            embed = hub.Module(module_url)

            stories = [' '.join(row[2:6]) for row in sent_arr]
            with tf.Session() as session:
                session.run([tf.global_variables_initializer(), tf.tables_initializer()])
                story_embeddings = session.run(embed(stories))

            df = pd.DataFrame(data=story_embeddings)
            df.to_csv(usc_embedding_path, sep=' ', header=None, index=False)

        self.story_embeddings = np.c_[
            pd.read_csv(usc_embedding_path, sep=' ', header=None).values,
            np.load(pathjoin(DATA_PATH, 'nlp_features.npy'))
        ]
        print(self.story_embeddings.shape)
        exit(-1)

        print('Computing closest')
        self.n_samples = len(sent_arr)
        with multiprocessing.Pool(4) as p:
            self.closest = p.map(self._closest_for_i, range(self.n_samples))
        self.closest = np.array(self.closest)

    def _closest_for_i(self, i):
        sim = np.full(self.n_samples, -np.inf)
        closest = -1
        for j in range(self.n_samples):
            if i == j:
                continue
            dot = np.dot(self.story_embeddings[i], self.story_embeddings[j])
            if dot > sim[i]:
                sim[i] = dot
                closest = j
        return closest

    def aug(self, i, row):
        row[6] = self.sent_arr[self.closest[i]][6]
        return row


def main():
    df = pd.read_csv(TRAIN_PATH)
    sent_arr = df.values
    X_neg = []
    #aug = AugBackward()
    #aug = AugRand(sent_arr)
    #aug = AugNearestLastSkipthought(sent_arr)
    #aug = AugNearestStorySkipthought(sent_arr)
    #aug = AugNearestAvgNumberbatch(sent_arr)
    #aug = AugNearestBertHiddenState(sent_arr)
    #aug = AugNearestStoryUSCEncoding(sent_arr)
    aug = AugNearestStoryUSCWithNLPEncoding(sent_arr)
    for i, row in enumerate(sent_arr):
        X_neg.append(aug.aug(i, row))
    X_neg = np.array(X_neg)
    df_neg = pd.DataFrame(index=df.index, columns=df.columns, data=X_neg)
    df_neg.to_csv(TRAIN_NEG_PATH, index=False)


if __name__ == '__main__':
    main()
