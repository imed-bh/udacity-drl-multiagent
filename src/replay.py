from collections import deque
import random
import torch

import numpy as np


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Experience:
    def __init__(self, state, action, reward, next_state, done):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, experience: Experience):
        self.buffer.append(experience)

    def sample(self):
        batch = random.sample(self.buffer, self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in batch])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in batch])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in batch])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in batch])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in batch]).astype(np.uint8)).float().to(device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)
