from torch.utils import data
import numpy as np


class Dataset(data.Dataset):
    def __init__(self, relations, labels, questions_lookup, answers_lookup):
        'Initialization'
        self.relations = relations
        self.labels = labels
        self.questions_lookup = questions_lookup
        self.answers_lookup = answers_lookup

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.relations)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.relations[index]

        # Load data and get label
        ques = self.questions_lookup[ID[0]]
        ans = self.answers_lookup[ID[1]]
        y = self.labels[ID]

        # convert to numpy array
        ques = np.array(ques)
        ans = np.array(ans)
        y = np.array(y).astype(np.float32)

        return ques, ans, y
