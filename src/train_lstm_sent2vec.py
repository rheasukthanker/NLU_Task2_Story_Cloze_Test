import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from os.path import join as pathjoin
import os
import subprocess

np.random.seed(42)
tf.set_random_seed(42)

N_SEQ_LEN = 5
DATA_PATH = 'data'
BATCH_SIZE = 32
TRAIN_PATH = pathjoin(DATA_PATH, 'cloze_test_val__spring2016 - cloze_test_ALL_val.csv')
TEST_PATH = pathjoin(DATA_PATH, 'test_for_report-stories_labels.csv')
TRAIN_EMBEDDINGS_PATH = pathjoin(DATA_PATH, 'train_sent2vec_embeddings.txt')
TEST_EMBEDDINGS_PATH = pathjoin(DATA_PATH, 'test_sent2vec_embeddings.txt')
MODEL_PATH = pathjoin(DATA_PATH, 'model_{epoch:02d}_{val_loss:.2f}.hdf5')
EMBEDDING_DIM = -1  # whatever

def gen_sent2vec_embeddings(dataset_path, embeddings_path):
    if os.path.exists(embeddings_path):
        return

    df = pd.read_csv(dataset_path)

    correct_id = df.values[:, -1].astype(np.int32) - 1
    wrong_id = 1 - correct_id
    n_samples = df.values.shape[0]
    X_pos = np.c_[df.values[:, 1:5], df.values[:, 5:7][np.arange(n_samples), correct_id]]
    X_neg = np.c_[df.values[:, 1:5], df.values[:, 5:7][np.arange(n_samples), wrong_id]]
    X = np.r_[X_pos, X_neg]

    sentence_path = pathjoin(DATA_PATH, 'sentences_oneline.txt')
    with open(sentence_path, 'w') as f:
        for sample in X:
            for sent in sample:
                f.write('{}\n'.format(sent))
    ps = subprocess.Popen(['cat', sentence_path], stdout=subprocess.PIPE)
    with open(embeddings_path, 'w') as f:
        subprocess.run(['sent2vec/fasttext', 'print-sentence-vectors',
            'data/torontobooks_unigrams.bin'],
            stdin=ps.stdout,
            stdout=f)
    os.remove(sentence_path)

def load_embeddings(dataset_path, embeddings_path):
    gen_sent2vec_embeddings(dataset_path, embeddings_path)
    with open(embeddings_path, 'r') as f:
        X = np.loadtxt(f)
    n_samples = X.shape[0] // N_SEQ_LEN
    X = X.reshape(n_samples, N_SEQ_LEN, EMBEDDING_DIM)
    y = np.r_[np.ones(n_samples // 2, dtype=np.int32), np.zeros(n_samples // 2, dtype=np.int32)]
    return X, y

X_train, y_train = load_embeddings(TRAIN_PATH, TRAIN_EMBEDDINGS_PATH)
X_test, y_test = load_embeddings(TEST_PATH, TEST_EMBEDDINGS_PATH)

model = keras.Sequential()
model.add(
    keras.layers.Bidirectional(
        keras.layers.LSTM(256,
                          return_sequences=True,
                          input_shape=[N_SEQ_LEN, X_train.shape[2]])))
model.add(keras.layers.Dropout(0.2))
model.add(
    keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True)))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Bidirectional(keras.layers.LSTM(64)))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(1, activation='sigmoid'))
model.add(keras.layers.Dropout(0.1))

adam = keras.optimizers.Adam(lr=5e-4)
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
model.fit(X_train,
          y_train,
          shuffle=True,
          batch_size=BATCH_SIZE,
          epochs=7,
          verbose=2)

n_test_samples = X_test.shape[0] // 2
correct_probs = model.predict(X_test).flatten()
np.save('correct_probs.npy', correct_probs)
higher_probs = correct_probs[:n_test_samples] > correct_probs[n_test_samples:]
accuracy = np.sum(higher_probs) / n_test_samples
print(f'Accuracy: {accuracy:.3f}')
