from utils.util import Preprocesser


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
    relation files (mapping of question and answer pairs to corresponding label):
        - train.pkl
        - dev.pkl
        - test.pkl
    mapping question and answer keys to preprocessed list of word indexes:
        - questions.pkl
        - answers.pkl
    mapping of words to indexes:
        - vocab.pkl
    matrix with weights saved at indexes of corresponding word:
        - embed.npz
        - norm_embed.npz (normalized version)
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
               q_len=2, a_len=10, stopwords=True)
    print('preprocessing is complete.')
