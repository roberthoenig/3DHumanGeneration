import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch


class ConditionalLinear(nn.Module):
    def __init__(self, num_in, num_out, n_steps):
        super(ConditionalLinear, self).__init__()
        self.num_out = num_out
        self.lin = nn.Linear(num_in, num_out)
        self.embed = nn.Embedding(n_steps, num_out)
        self.embed.weight.data.uniform_()

    def forward(self, x, y):
        out = self.lin(x)
        gamma = self.embed(y)
        out = gamma.view(-1, self.num_out) * out
        return out


class ConditionalModel(nn.Module):
    def __init__(self, n_steps, in_sz, cond_sz, cond_model):
        super(ConditionalModel, self).__init__()
        self.cond_model = cond_model
        self.lin1 = ConditionalLinear(in_sz+cond_sz, 128, n_steps)
        self.lin2 = ConditionalLinear(128, 128, n_steps)
        self.lin3 = ConditionalLinear(128, 128, n_steps)
        self.lin4 = nn.Linear(128, in_sz)

    def forward(self, x, y, cond):
        cond_processed = self.cond_model(cond)
        x = torch.cat([x, cond_processed], dim=-1)
        x = F.softplus(self.lin1(x, y))
        x = F.softplus(self.lin2(x, y))
        x = F.softplus(self.lin3(x, y))
        return self.lin4(x)
