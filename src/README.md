# Data Augmentation to Improve BERT on Story Cloze
For required packages, see requirements.txt.

## Data Exploration Notebook
A jupyter notebook analyzing the distribution of various features on the right and the wrong endings of the Story Cloze Task

## Classical NLP Logistic Regression
### Prerequisities
 Additionally, to run classical NLP
part, you need to run
```
nltk.download('words')
nltk.download('maxent_ne_chunker')
```
To run the logistic regression model use ```python classical_nlp.py```

## Bidirectional LSTM
In order to run BiLSTM, you need to download sent2vec and compile it. To do this run the following commands.
1. `git clone https://github.com/epfml/sent2vec.git sent2vec`
2. `cd sent2vec`
3. `make`

Afterwards, simply run
```
python train_lstm_sent2vec.py
```
to train the described BiLSTM model on Story Cloze validation set and test it on Story Cloze
test set.

## BERT
All of our BERT codes are in separate notebooks for each experiment. Each notebook can be found in `bert_notebooks` folder.
In addition to common packages, BERT notebooks require `bert-tensorflow` module. However, they automatically
install this module when they are run.

## Data Augmentation by Choosing Endings
Data augmentation codes are in `dataset_augmentation.py` file. SkipThought augmentation strategies
require sent2vec to be built in the same way as BiLSTM section. In addition, USC+NLP strategy requires
`nlp_features.npy` file to be in `data` folder. If you want to use this strategy, generate the NLP features by running
`python gen_nlp_features.py`.

## Data Augmentation by Generating Endings
To generate endings, we need to GPT-2 repository. Follow these steps:
1. Clone the repository: `git clone https://github.com/openai/gpt-2.git gpt-2`
2. `cd gpt-2`
3. Install GPT-2 requirements: `pip install -r requirements.txt`
4. Download GPT-2 model: `python download_model.py 117M`
5. Set these parameters in `src/interactive_conditional_samples.py` file:
  1. `seed=1`
  2. `top_k=40`
  3. `length=25`
6. Run `gpt_run.sh` to generate GPT-2 samples for each story.
