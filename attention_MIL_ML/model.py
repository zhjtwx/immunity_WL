import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.L = 1454
        self.D = 256
        self.K = 1
        self.ReLU = nn.ReLU(inplace=True)
        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.L * self.K, self.L * self.K // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.L * self.K // 2, self.L * self.K // 4),
            nn.Dropout(0.2),
            nn.ReLU(inplace=True),
            nn.Linear(self.L * self.K // 4, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, H):
        A = self.attention(H)  # NxK
        A = torch.transpose(A, 2, 1)  # KxN
        b, c = A.size()[0], A.size()[2]
        for b_i in range(b):
            for c_i in range(c):
                if torch.sum(H[b_i, c_i, :]) == 0:
                    A[b_i, 0, c_i] = 0.
        A = F.softmax(A, dim=2)  # softmax over N
        for b_i in range(b):
            for c_i in range(c):
                if torch.sum(H[b_i, c_i, :]) == 0:
                    A[b_i, 0, c_i] = 0.
        # print(A.size())
        for b_i in range(b):
            A[b_i] = torch.div(A[b_i], torch.sum(A[b_i] + 0.000000001))
        # print(A.size())
        M = torch.FloatTensor([torch.mm(A[i], H[i]).detach().numpy() for i in range(A.size()[0])])
        M = M.view(-1, 1454)

        Y_prob = self.classifier(M)
        # print(Y_prob, A)
        return (Y_prob, M), A
