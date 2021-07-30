#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 08:06:35 2019

@author: rheasukthanker
"""
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from textblob import TextBlob
import nltk
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from nltk.sentiment.vader import SentimentIntensityAnalyzer
#nltk.download("words")
#nltk.download("maxent_ne_chunker")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords


#val_data=pd.read_csv("data/cloze_test_val__spring2016 - cloze_test_ALL_val.csv")
def get_1gram_intersection_count(endings, passages):
    intersection_count = []
    for i in range(np.shape(passages)[0]):
        ending_tokenized = nltk.word_tokenize(endings[i].lower())
        intersection_count.append(
            len(list(set(ending_tokenized) & set(passages[i]))))
    return np.reshape(np.array(intersection_count), [len(endings), 1])


def get_polarity_difference(passages, endings):
    polarity_diff = []
    subjectivity_diff = []
    for i in range(0, np.shape(passages)[0]):
        passage_polarity = TextBlob(passages[i]).sentiment
        ending_polarity = TextBlob(endings[i]).sentiment
        polarity_diff.append(np.abs(passage_polarity[0] - ending_polarity[0]))
        subjectivity_diff.append(
            np.abs(passage_polarity[1] - ending_polarity[1]))
    return np.reshape(np.array(polarity_diff),
                      [len(polarity_diff), 1]), np.reshape(
                          np.array(subjectivity_diff),
                          [len(subjectivity_diff), 1])


def remove_stopwords(tokenized_sent):
    stops = set(stopwords.words('english'))
    new_sent = []
    for x in tokenized_sent:
        if x not in stops:
            new_sent.append(x)
    return new_sent


def get_1gram_intersection_count_duplicates(endings, passages):
    intersection_count = []
    for i in range(np.shape(passages)[0]):
        ending_tokenized = nltk.word_tokenize(endings[i].lower())
        count = 0
        for x in ending_tokenized:
            if x in passages[i]:
                count = count + 1
        #print(count)
        intersection_count.append(count)
    return np.reshape(np.array(intersection_count), [len(endings), 1])


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


def get_word_ngrams(endings):
    #set range to test effect of changing the ngram range
    word_vectorizer = TfidfVectorizer(sublinear_tf=True,
                                      strip_accents='unicode',
                                      analyzer='word',
                                      token_pattern=r'\w{1,}',
                                      stop_words='english',
                                      ngram_range=(2, 5),
                                      max_features=20000)
    word_vectorizer.fit(endings)
    train_word_features = word_vectorizer.transform(endings)
    return np.array(train_word_features.todense())


def get_char_ngrams(endings):
    char_vectorizer = TfidfVectorizer(sublinear_tf=True,
                                      strip_accents='unicode',
                                      analyzer='char',
                                      stop_words='english',
                                      ngram_range=(4, 9),
                                      max_features=35000)
    char_vectorizer.fit(endings)
    train_char_features = char_vectorizer.transform(endings)
    return np.array(train_char_features.todense())


def get_pca(data, keep):
    pca = PCA(n_components=keep, svd_solver='full')
    pca.fit(data)
    data_new = pca.transform(data)
    return data_new


def vader(endings):
    neg = []
    neu = []
    pos = []
    compound = []
    sid = SentimentIntensityAnalyzer()
    for x in endings:
        ss = sid.polarity_scores(x)
        neg.append(ss['neg'])
        neu.append(ss['neu'])
        pos.append(ss['pos'])
        compound.append(ss['compound'])
    return np.array(neg), np.array(neu), np.array(pos), np.array(compound)


def shuffle_in_unision(data, labels):
    c = np.c_[data, np.array(labels).reshape(len(labels), -1)]
    np.random.shuffle(c)
    data_shuffled = c[:, :data.size // len(data)].reshape(data.shape)
    labels_shuffled = c[:, data.size // len(data):].reshape(
        np.array(labels).shape)
    return data_shuffled, labels_shuffled


def get_cross_val_score(data, labels, clf):
    scores = cross_val_score(clf, data, labels, cv=5)
    print("5-fold CV scores", scores)
    return np.mean(scores), np.std(scores)


def document_features(endings):
    token = RegexpTokenizer(r'[a-zA-Z0-9]+')
    cv = CountVectorizer(lowercase=True,
                         stop_words='english',
                         ngram_range=(1, 1),
                         tokenizer=token.tokenize)
    text_counts = cv.fit_transform(endings)
    #print(np.shape(text_counts))
    return text_counts.todense()


def main():
    val_data = pd.read_csv(
        "data/cloze_test_val__spring2016 - cloze_test_ALL_val.csv")
    labels = []
    endings = []
    passages = []
    passages_untok = []
    for i in range(0, val_data.shape[0]):
        sent1 = nltk.word_tokenize(val_data.iloc[i, 1].lower())
        sent2 = nltk.word_tokenize(val_data.iloc[i, 2].lower())
        sent3 = nltk.word_tokenize(val_data.iloc[i, 3].lower())
        sent4 = nltk.word_tokenize(val_data.iloc[i, 4].lower())
        passages_untok.append(val_data.iloc[i, 1].lower() + " " +
                              val_data.iloc[i, 2].lower() + " " +
                              val_data.iloc[i, 3].lower() + " " +
                              val_data.iloc[i, 4].lower())
        passages_untok.append(val_data.iloc[i, 1].lower() + " " +
                              val_data.iloc[i, 2].lower() + " " +
                              val_data.iloc[i, 3].lower() + " " +
                              val_data.iloc[i, 4].lower())
        doc = [sent1, sent2, sent3, sent4]
        doc_unlist = [item for sublist in doc for item in sublist]
        passages.append(doc_unlist)
        passages.append(doc_unlist)
        lab = val_data.iloc[i, 7]
        end1 = val_data.iloc[i, 5]
        end2 = val_data.iloc[i, 6]
        endings.append(end1)
        endings.append(end2)
        if lab == 1:
            labels.append(1)
        else:
            labels.append(0)
        if lab == 2:
            labels.append(1)
        else:
            labels.append(0)
    #labels=np.array(labels)
    word_features = get_word_ngrams(endings)
    char_features = get_char_ngrams(endings)
    lengths = np.reshape(get_length(endings), [len(endings), 1])
    doc_features = document_features(endings)
    polarity, subjectivity = get_pol_sub(endings)
    ner_features = get_ner_counts(endings)
    neg, neu, pos, compound = vader(endings)
    neg = np.reshape(neg, [len(endings), 1])
    neu = np.reshape(neu, [len(endings), 1])
    pos = np.reshape(pos, [len(endings), 1])
    compound = np.reshape(compound, [len(endings), 1])
    #full_data=np.concatenate([word_features,char_features,lengths,ner_features,polarity,subjectivity],axis=1)
    #Shuffle
    #print(doc_)
    #print(endings[1])
    #print(passages[1])
    one_gram_duplicates = get_1gram_intersection_count_duplicates(
        endings, passages)
    one_gram_count = get_1gram_intersection_count(endings, passages)
    polarity_diff, subjectivity_diff = get_polarity_difference(
        passages_untok, endings)
    #print(one_gram_count[1])
    #print(one_gram_duplicates)
    #print(one_gram_count)
    full_data = np.concatenate([
        lengths, word_features, char_features, neg, neu, pos, compound,
        ner_features, one_gram_count, polarity_diff, subjectivity_diff,
        polarity, subjectivity
    ],
                               axis=1)
    full_data, labels = shuffle_in_unision(full_data, labels)
    labels = labels.tolist()
    clf = LogisticRegression(random_state=0, solver='lbfgs', max_iter=1000000)
    print("Mean and Standard Deviation CV",
          get_cross_val_score(full_data, labels, clf))
    clf = LogisticRegression(random_state=0, solver='lbfgs', max_iter=1000000)

    clf.fit(full_data, labels)
    test_data = pd.read_csv(
        "data/cloze_test_test__spring2016 - cloze_test_ALL_test.csv")
    labels = []
    lab_act = []
    endings = []
    passages = []
    passages_untok = []
    for i in range(0, test_data.shape[0]):
        sent1 = nltk.word_tokenize(test_data.iloc[i, 1].lower())
        sent2 = nltk.word_tokenize(test_data.iloc[i, 2].lower())
        sent3 = nltk.word_tokenize(test_data.iloc[i, 3].lower())
        sent4 = nltk.word_tokenize(test_data.iloc[i, 4].lower())
        passages_untok.append(test_data.iloc[i, 1].lower() + " " +
                              test_data.iloc[i, 2].lower() + " " +
                              test_data.iloc[i, 3].lower() + " " +
                              test_data.iloc[i, 4].lower())
        passages_untok.append(test_data.iloc[i, 1].lower() + " " +
                              test_data.iloc[i, 2].lower() + " " +
                              test_data.iloc[i, 3].lower() + " " +
                              test_data.iloc[i, 4].lower())
        doc = [sent1, sent2, sent3, sent4]
        doc_unlist = [item for sublist in doc for item in sublist]
        passages.append(doc_unlist)
        passages.append(doc_unlist)
        lab = test_data.iloc[i, 7]
        lab_act.append(lab)
        end1 = test_data.iloc[i, 5]
        end2 = test_data.iloc[i, 6]
        endings.append(end1)
        endings.append(end2)
        if lab == 1:
            labels.append(1)
        else:
            labels.append(0)
        if lab == 2:
            labels.append(1)
        else:
            labels.append(0)
    #labels=np.array(labels)
    word_features = get_word_ngrams(endings)
    char_features = get_char_ngrams(endings)
    lengths = np.reshape(get_length(endings), [len(endings), 1])
    doc_features = document_features(endings)
    polarity, subjectivity = get_pol_sub(endings)
    ner_features = get_ner_counts(endings)
    neg, neu, pos, compound = vader(endings)
    neg = np.reshape(neg, [len(endings), 1])
    neu = np.reshape(neu, [len(endings), 1])
    pos = np.reshape(pos, [len(endings), 1])
    compound = np.reshape(compound, [len(endings), 1])
    #full_data=np.concatenate([word_features,char_features,lengths,ner_features,polarity,subjectivity],axis=1)
    #Shuffle
    #print(doc_)
    #print(endings[1])
    #print(passages[1])
    one_gram_duplicates = get_1gram_intersection_count_duplicates(
        endings, passages)
    one_gram_count = get_1gram_intersection_count(endings, passages)
    polarity_diff, subjectivity_diff = get_polarity_difference(
        passages_untok, endings)
    #print(one_gram_count[1])
    #print(one_gram_duplicates)
    #print(one_gram_count)
    full_data = np.concatenate([
        lengths, word_features, char_features, neg, neu, pos, compound,
        ner_features, one_gram_count, polarity_diff, subjectivity_diff,
        polarity, subjectivity
    ],
                               axis=1)
    preds = clf.predict(full_data)
    print("Test accuracy on classification task",
          accuracy_score(labels, preds))
    probs = clf.predict_proba(full_data)
    #print(labels)
    prob_corr = probs[:, 1]
    argmax_list = []
    i = 0
    while i < np.shape(prob_corr)[0] - 2:
        argmax_list.append([prob_corr[i], prob_corr[i + 2]])
        i = i + 2
    #print(probs)
    am = np.array(argmax_list)
    maxi = np.argmax(am, axis=1) + 1
    correct_count = 0
    for i in range(0, np.shape(maxi)[0]):
        if lab_act[i] == maxi[i]:
            correct_count = correct_count + 1
    print("Test accuracy after argmax:", correct_count / np.shape(maxi)[0])


if __name__ == "__main__":
    main()
