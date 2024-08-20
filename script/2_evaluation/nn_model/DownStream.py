import os, sys
import torch 
import torch.nn as nn
import torch.nn.functional as F



class Corse(nn.Module):
    def __init__(self, in_dim=512, out_dim=2):
        super().__init__()
        self.fn = nn.Sequential(
            nn.Linear(in_dim, in_dim * 2),
            # nn.ReLU(),
            # nn.Linear(in_dim * 2, in_dim * 2),
        )
        self.ln = nn.Linear(in_dim * 2, out_dim)

    def forward(self, x, softmax=False):
        logits = self.fn(x)
        logits, _ = torch.max(self.ln(F.relu(logits)), dim=1)
        if softmax:
            logits = F.softmax(x)
        return logits
    
class Fine(nn.Module):
    def __init__(self, in_dim=512, out_dim=512):
        super().__init__()
        self.fn = nn.Sequential(
            nn.Linear(in_dim, in_dim * 2),
        )
        self.ln = nn.Linear(in_dim * 2, 1)

    def forward(self, x, softmax=False):
        logits = self.ln(F.relu(self.fn(x)))
        logits = torch.squeeze(logits)
        if softmax:
            logits = F.softmax(x)
        return logits
    
class TextCNN(nn.Module):
    def __init__(self, in_dim=512, kernel_sizes=[1,2,3,4,5], out_dim=512):
        super().__init__()

        self.model = nn.ModuleList(
            [nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=k) for k in kernel_sizes]
        )
        self.dropout = nn.Dropout()
        self.fc = nn.Linear(len(kernel_sizes) * in_dim, out_dim)

    def forward(self, x:torch.Tensor):
            # print(x.shape)
            x = x.transpose(1, 2)
            x = [F.relu(conv(x)) for conv in self.model]
            x = [F.max_pool1d(c, c.size(-1)).squeeze(dim=-1) for c in x]
            x = torch.concat(x, dim=1)
            # print(x.shape)
            x = self.fc(self.dropout(x))
            return x.squeeze()