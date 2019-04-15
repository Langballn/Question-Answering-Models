import csv
import re
import string
import pickle
import numpy as np
from nltk.corpus import stopwords


class Preprocesser:
    def __init__(self, q_len, a_len, use_stopwords=True, use_padding=True):
        self.q_len = q_len  # maximum question length
        self.a_len = a_len  # maximum answer length
        self.stop_words = set(stopwords.words(
            'english')) if use_stopwords else None
        self.regex = re.compile(
            '[%s]' % re.escape(string.punctuation))  # remove punctuation
        self.use_padding = use_padding  # option to pad sequences to max length
        self.vocab = {}  # vocabulary dictonary - assign each word to a number in order to index embedding vectors
        self.vocab_len = 1 if use_padding else 0  # vocabulary length
        self.answers = {}
        self.questions = {}
        self.relations = {}
        self.labels = {}

    def _preprocess_text(self, text):
        # function to preprocess string
        # make all letters lower case, remove punctuation and remove stopwords
        text = text.lower()
        text = self.regex.sub('', text)
        tokens = [token for token in text.split(
        ) if token not in self.stop_words]
        # add preprocessed tokens to the vocabulary
        for token in tokens:
            if token not in self.vocab:
                self.vocab[token] = self.vocab_len
                self.vocab_len += 1
        # convert words to idx values
        return [self.vocab[token] for token in tokens]

    def _preprocess_file(self, in_filename, set_name):
        # read in raw data file with each sample formatted as: question /t answer /t label /n
        # output preprocessed file with words referenced by numbers in vocab
        with open(in_filename, 'r') as in_file:
            for row in in_file:
                line = row.strip().split('\t')
                # preprocess question and answers
                question = self._preprocess_text(line[0])[: self.q_len]
                answer = self._preprocess_text(line[1])[: self.a_len]
                label = int(line[2])

                # pad sequences if less than max length
                if self.use_padding:
                    q_padding = (self.q_len - len(question)) * [0]
                    question.extend(q_padding)
                    a_padding = (self.a_len - len(answer)) * [0]
                    answer.extend(a_padding)

                # save question, answer and corresponding label
                q_key = 'Q' + str(len(self.questions))
                a_key = 'A' + str(len(self.answers))
                self.answers[a_key] = answer
                self.questions[q_key] = question
                self.labels[(q_key, a_key)] = label
                if set_name not in self.relations:
                    self.relations[set_name] = []
                self.relations[set_name].append((q_key, a_key))

    def _load_embeddings(self, embed_file):
        # read in raw embeddings file formatted as: word /s vector /n
        embeddings = None
        with open(embed_file, 'r') as file:
            for row in file:
                line = row.split()
                word = line[0]

                if word in self.vocab:
                    vector = np.array(line[1:]).astype(float)
                    idx = self.vocab[word]
                    # initialize embedding weights to be a matrix of size vocab length and embedding dimension
                    # use randomly initialized values in case the pretrained embeddings do not contain the entire vocabulary
                    if embeddings is None:
                        embeddings = np.random.rand(
                            self.vocab_len, len(vector))
                        embeddings[0] = np.zeros(len(vector))
                    # save pretrained weights vector at word index
                    embeddings[idx] = vector

        # normalize embeddings
        (rows, cols) = embeddings.shape
        magnitude = np.sqrt(
            np.sum(np.multiply(embeddings, embeddings), axis=1)).reshape(rows, 1)
        norm_embed = embeddings / magnitude

        if self.use_padding:
            pad = np.zeros((cols, ))
            embeddings[0] = pad
            norm_embed[0] = pad

        np.save('data/embed', embeddings)
        np.save('data/norm_embed', norm_embed)

    def preprocess(self, train_file, dev_file, test_file, embed_file):
        # preprocess test, dev and train sets and save to pkl
        self._preprocess_file(train_file, 'train')
        self._preprocess_file(dev_file, 'dev')
        self._preprocess_file(test_file, 'test')

        # save relations to pkl
        f = open('data/relations.pkl', 'wb')
        pickle.dump(self.relations, f)
        f.close()

        # save labels to pkl
        f = open('data/labels.pkl', 'wb')
        pickle.dump(self.labels, f)
        f.close()

        # save questions to pkl
        f = open('data/questions.pkl', 'wb')
        pickle.dump(self.questions, f)
        f.close()

        # save answers to pkl
        f = open('data/answers.pkl', 'wb')
        pickle.dump(self.answers, f)
        f.close()

        # save vocabulary to pkl
        f = open('data/vocab.pkl', 'wb')
        pickle.dump(self.vocab, f)
        f.close()

        # preprocess embeddings and to npz file
        self._load_embeddings(embed_file)
