import numpy as np

from src.agent import Agent
from src.tennis_env import TennisEnv


class RandomAgent(Agent):
    def __init__(self, env: TennisEnv):
        super().__init__(env)

    def compute_action(self, state, epsilon):
        return np.clip(np.random.randn(2*self.env.action_size)*0.2, -1, 1)
