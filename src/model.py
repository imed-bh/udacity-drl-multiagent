import torch
from torch import nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


INIT_WEIGHTS = 1e-6


class Actor(nn.Module):
    def __init__(self, state_size, action_size, fc1_units, fc2_units):
        super().__init__()
        self.fc1 = nn.Linear(state_size, fc1_units).to(device)
        self.fc2 = nn.Linear(fc1_units, fc2_units).to(device)
        self.fc3 = nn.Linear(fc2_units, action_size).to(device)
        self.fc1.weight.data.uniform_(-INIT_WEIGHTS, INIT_WEIGHTS)
        self.fc2.weight.data.uniform_(-INIT_WEIGHTS, INIT_WEIGHTS)
        self.fc3.weight.data.uniform_(-INIT_WEIGHTS, INIT_WEIGHTS)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))

    def action_values_for(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.eval()
        with torch.no_grad():
            action_values = self(state)
        self.train()
        return action_values.cpu().data.numpy()

    def soft_update(self, other, tau):
        for self_param, other_param in zip(self.parameters(), other.parameters()):
            self_param.data.copy_(tau * other_param.data + (1 - tau) * self_param.data)


class Critic(nn.Module):
    def __init__(self, state_size, action_size, fc1_units, fc2_units):
        super().__init__()
        self.fc1 = nn.Linear(state_size+action_size, fc1_units).to(device)
        self.fc2 = nn.Linear(fc1_units, fc2_units).to(device)
        self.fc3 = nn.Linear(fc2_units, 1).to(device)
        self.fc1.weight.data.uniform_(-INIT_WEIGHTS, INIT_WEIGHTS)
        self.fc2.weight.data.uniform_(-INIT_WEIGHTS, INIT_WEIGHTS)
        self.fc3.weight.data.uniform_(-INIT_WEIGHTS, INIT_WEIGHTS)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def soft_update(self, other, tau):
        for self_param, other_param in zip(self.parameters(), other.parameters()):
            self_param.data.copy_(tau * other_param.data + (1 - tau) * self_param.data)
