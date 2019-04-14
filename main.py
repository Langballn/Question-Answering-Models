from utils.util import Preprocesser


def preprocess(train_file, dev_file, test_file, embed_file, q_len=None, a_len=None, stopwords=True):
    '''
    Reads in pretrained embeddings, train, development, and test data then preprocesses and serializes results

    Parameters:
    train_file: path of training data
    dev_file: path of development data
    test_file: path of test data
    embed_file: path of embeddings
    q_len (int): maximum question length (default is None)
    a_len (int): maximum answer length (default is None)
    stopwords (bool): whether to remove stop words (default is True)

    Returns:
    relation files (mapping of question and answer pairs to corresponding label):
        - train.pkl
        - dev.pkl
        - test.pkl
    mapping question and answer keys to preprocessed list of word indexes:
        - questions.pkl
        - answers.pkl
    mapping of words to indexes:
        - vocab.pkl
    embeddings with weights saved at indexes of corresponding word:
        - embeddings.npz
    '''

    preprocesser = Preprocesser(q_len, a_len, stopwords)
    preprocesser.preprocess(train_file, dev_file, test_file, embed_file)


if __name__ == '__main__':
    # specify location of dataset and embeddings
    train_file = 'data/wikiqa/WikiQA-train.txt'
    dev_file = 'data/wikiqa/WikiQA-dev.txt'
    test_file = 'data/wikiqa/WikiQA-test.txt'
    embed_file = 'data/embed/glove.6B.300d.txt'

    print('preprocessing dataset and embeddings ....')
    preprocess(train_file, dev_file, test_file, embed_file,
               q_len=None, a_len=None, stopwords=True)
    print('preprocessing is complete.')