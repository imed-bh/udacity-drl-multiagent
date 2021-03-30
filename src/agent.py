from abc import ABC, abstractmethod
import numpy as np

from src.tennis_env import TennisEnv


class Agent(ABC):
    """
    This class is the base for the different agents. Currently there is only random agent and DDPG agent
    but the idea is to be able to extend and compare other types of agents.
    """
    def __init__(self, env: TennisEnv):
        self.env = env
        self.t_step = 0
        self.train_state = None

    @abstractmethod
    def compute_action(self, state, epsilon):
        pass

    def evaluate(self, n=10, fast=True):
        scores = [self.run_episode(fast) for _ in range(n)]
        return np.mean(scores)

    def run_episode(self, fast=False):
        state = self.env.reset(train_mode=fast)
        score = 0
        done = False
        while not done:
            action = self.compute_action(state, epsilon=0)
            state, reward, done = self.env.step(action)
            score += reward
            if reward != 0 and not fast:
                print(f"Score {score:.2f}")
        return score
