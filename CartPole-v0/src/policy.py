import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical

if torch.cuda.is_available():
    FLOAT = torch.FloatTensor
else:
    FLOAT = torch.cuda.FloatTensor
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Policy(nn.Module):
    def __init__(self, state_space, n_actions):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_space, 32, bias=False)
        self.fc2 = nn.Linear(32, n_actions, bias=False)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.softmax(x)
        return x

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        state = self.forward(Variable(state))
        c = Categorical(state)
        action = c.sample()
        return action.item(), c.log_prob(action)
