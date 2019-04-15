import pickle


def load_pkl(filename):
    f = open(filename, 'rb')
    data = pickle.load(f)
    f.close()
    return data
