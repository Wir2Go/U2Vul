import os, sys
import torch 
import torch.nn as nn
import torch.nn.functional as F

class Corse(nn.Module):
    def __init__(self, in_dim=512, out_dim=2):
        super().__init__()
        self.fn = nn.Sequential(
            nn.Linear(in_dim, in_dim * 2),
            nn.ReLU(),
            nn.Linear(in_dim * 2, in_dim * 2),
        )
        self.ln = nn.Linear(in_dim * 2, out_dim)

    def forward(self, x, softmax=False):
        logits = self.fn(x)
        logits, _ = torch.max(self.ln(F.relu(logits)), dim=1)
        if softmax:
            logits = F.softmax(x)
        return logits
    
class Fine(nn.Module):
    def __init__(self, in_dim=512):
        super().__init__()
        self.fn = nn.Sequential(
            nn.Linear(in_dim, in_dim * 2),
            nn.ReLU(),
            nn.Linear(in_dim * 2, in_dim * 2),
        )
        self.ln = nn.Linear(in_dim * 2, 1)

    def forward(self, x, softmax=False):
        logits = self.ln(self.fn(x))
        logits = torch.squeeze(logits)
        if softmax:
            logits = F.softmax(x)
        return logits