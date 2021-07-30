import numpy as np
import pandas as pd
from textblob import TextBlob
import os
import nltk
#nltk.download("words")
#nltk.download("maxent_ne_chunker")

def get_length(endings):
    return [len(nltk.word_tokenize(i)) for i in endings]

def get_pol_sub(endings):
    polarity = []
    subjectivity = []
    for x in endings:
        polarity.append(TextBlob(x).sentiment[0])
        subjectivity.append(TextBlob(x).sentiment[1])
    polarity = np.reshape(polarity, [len(endings), 1])
    subjectivity = np.reshape(subjectivity, [len(endings), 1])
    return polarity, subjectivity


def get_ner_counts(endings):
    endings_ner = []
    for sent in endings:
        ending_ner = []
        for sent1 in nltk.sent_tokenize(sent):
            #print(sent1)
            for chunk in nltk.ne_chunk(nltk.pos_tag(
                    nltk.word_tokenize(sent1))):
                if hasattr(chunk, 'label'):
                    # print(chunk.label(), ' '.join(c[0] for c in chunk))
                    ending_ner.append(chunk.label())
        endings_ner.append(ending_ner)
    features_ner = []
    for i in endings_ner:
        if len(i) > 0:
            org_count = 0
            pers_count = 0
            gpe_count = 0
            for x in i:
                if x == "ORGANIZATION":
                    org_count += 1
                elif x == "PERSON":
                    pers_count += 1
                elif x == "GPE":
                    gpe_count += 1
            features_ner.append([org_count, pers_count, gpe_count])
        else:
            features_ner.append([0, 0, 0])
    return np.array(features_ner)


def main():
    outpath = 'data/nlp_features.npy'
    if not os.path.exists(outpath):
        path = 'data/train_stories.csv'
        df = pd.read_csv(path)
        endings = df.values[:, 6]
        lengths = np.reshape(get_length(endings), [len(endings), 1])
        polarity, subjectivity = get_pol_sub(endings)
        ner_features = get_ner_counts(endings)
        np.save(outpath, np.c_[lengths, polarity, subjectivity, ner_features])


if __name__ == "__main__":
    main()
