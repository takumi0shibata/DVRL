import torch
import torch.nn as nn

class SoftAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SoftAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.w = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.v = nn.Linear(self.hidden_dim, 1)

    def forward(self, h):
        w = torch.tanh(self.w(h))

        weight = self.v(w)
        weight = weight.squeeze(dim=-1)

        weight = torch.softmax(weight, dim=1)
        weight = weight.unsqueeze(dim=-1)
        out = torch.mul(h, weight.repeat(1, 1, h.size(2)))

        out = torch.sum(out, dim=1)

        return out