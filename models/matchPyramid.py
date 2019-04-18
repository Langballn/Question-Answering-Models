import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np


class MatchPyramid(nn.Module):
    def __init__(self, BATCH_SIZE, Q_LEN, A_LEN):
        super(MatchPyramid, self).__init__()
        # load embeddings
        self.embed = self.load_embeddings()
        # input channels is 1 and output channels is set to 18
        self.conv1 = torch.nn.Conv2d(1, 18, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = torch.nn.Linear(18 * 16 * 16, 64)
        self.fc2 = torch.nn.Linear(64, 10)
        self.fc3 = torch.nn.Linear(10, 1)
        self.BATCH_SIZE = BATCH_SIZE
        self.Q_LEN = Q_LEN
        self.A_LEN = A_LEN

    def load_embeddings(self):
        pretrained_weights = np.load('data/norm_embed.npy')
        embed = nn.Embedding(
            pretrained_weights.shape[0], pretrained_weights.shape[1])
        embed.weight.data.copy_(torch.from_numpy(pretrained_weights))
        embed.weight.requires_grad = False
        return embed

    def get_input_matrix(self, q_lookup, a_lookup):
        # returns a matrix with cosine similarity of the inputs
        ans = self.embed(torch.LongTensor(a_lookup))
        ques = self.embed(torch.LongTensor(q_lookup))
        input = torch.Tensor(self.BATCH_SIZE, 1, self.Q_LEN, self.A_LEN)
        for i, q in enumerate(ques):
            input[i][0] = torch.mm(q, ans[i].t())
        return input

    def forward(self, q, a):
        # get input matrix (cosine similarly bw question and answer embeddings)
        x = self.get_input_matrix(q, a)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(-1, 18 * 16 * 16)
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return(x)
