import numpy as np

from src.tennis_env import TennisEnv


class RandomAgent:
    def __init__(self, env: TennisEnv):
        self.env = env

    def compute_action(self, state, epsilon):
        return np.clip(np.random.randn(self.env.action_size)*0.2, -1, 1)
