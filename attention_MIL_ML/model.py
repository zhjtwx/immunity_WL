import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.L = 1454
        self.D = 256
        self.K = 1454
        self.ReLU = nn.ReLU(inplace=True)
        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            # nn.Linear(self.L, self.L // 2),
            # nn.ReLU(inplace=True),
            nn.Linear(self.L, self.L // 4),
            nn.Dropout(0.2),
            nn.ReLU(inplace=True),
            nn.Linear(self.L // 4, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, H):
        A = self.attention(H)  # NxK

        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.FloatTensor([torch.mul(A[i], H[i]).detach().numpy() for i in range(A.size()[0])])
        M = torch.sum(M, dim=1)
        M = M.view(-1, 1454)

        Y_prob = self.classifier(M)

        return Y_prob, A
