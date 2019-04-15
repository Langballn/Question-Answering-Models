from utils.preprocess import Preprocesser
from utils.dataloader import Dataset
from utils.util import load_pkl
from models.matchPyramid import MatchPyramid
import torch
import torch.nn as nn
from torch.utils import data


def preprocess(train_file, dev_file, test_file, embed_file, q_len=None, a_len=None, use_stopwords=True, use_padding=True):
    '''
    Reads in pretrained embeddings, train, development, and test data then preprocesses and serializes results

    Parameters:
    train_file: path of training data
    dev_file: path of development data
    test_file: path of test data
    embed_file: path of embeddings data
    q_len (int): maximum question length (default is None)
    a_len (int): maximum answer length (default is None)
    use_stopwords (bool): whether to remove stop words (default is True)
    use_padding (bool): whether to pad sequences (default is True)

    Returns:
    list of relations (question, answer) pairs belonging to each set (train, test, dev set)
        - relations.pkl
    mapping of (question, answer) pairs to corresponding label):
        - labels.pkl
    mapping question key to question (preprocessed list of word indices):
        - questions.pkl
    mapping answer key to answer (preprocessed list of word indices):
        - answers.pkl
    mapping of words to indexes:
        - vocab.pkl
    matrix with weights saved at indexes of corresponding word:
        - embed.npz
        - norm_embed.npz (normalized version)
    '''

    preprocesser = Preprocesser(q_len, a_len, use_stopwords, use_padding)
    preprocesser.preprocess(train_file, dev_file, test_file, embed_file)


def train():

    # Construct our model by instantiating the class
    model = MatchPyramid()

    # Create data iterator
    # Parameters
    params = {'batch_size': 64,
              'shuffle': True,
              'num_workers': 1}
    max_epochs = 100

    # Datasets
    relations = load_pkl('data/relations.pkl')
    labels = load_pkl('data/labels.pkl')
    answers_lookup = load_pkl('data/answers.pkl')
    questions_lookup = load_pkl('data/questions.pkl')

    # Generators
    training_set = Dataset(
        relations['train'], labels, questions_lookup, answers_lookup)
    training_generator = data.DataLoader(training_set, **params)

    validation_set = Dataset(
        relations['dev'], labels, questions_lookup, answers_lookup)
    validation_generator = data.DataLoader(validation_set, **params)

    # Construct our loss function and an Optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

    # Loop over epochs
    for epoch in range(max_epochs):
        # Training
        for local_batch_ques, local_batch_ans, local_labels in training_generator:

            # Forward pass: Compute predicted label based on question and answer
            pred_labels = model(local_batch_ques, local_batch_ans)

            # Compute and print loss
            loss = criterion(pred_labels, local_labels)
            print(epoch, loss.item())

            # Zero gradients, perform a backward pass, and update the weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


if __name__ == '__main__':
    # specify location of dataset and embeddings
    train_file = 'data/wikiqa/WikiQA-train.txt'
    dev_file = 'data/wikiqa/WikiQA-dev.txt'
    test_file = 'data/wikiqa/WikiQA-test.txt'
    embed_file = 'data/embed/glove.6B.300d.txt'

    # print('Preprocessing dataset and embeddings ....')
    # preprocess(train_file, dev_file, test_file, embed_file,
    #            q_len=32, a_len=32)
    # print('preprocessing is complete.')

    print('Training model ...')
    train()
