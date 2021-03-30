from collections import deque
import numpy as np
import matplotlib.pyplot as plt


class Metrics:
    """
    This class is used to keep track of the running average of score for the last 100 episodes
    and plot it in real-time to a matplotlib figure.
    """
    def __init__(self):
        self.step_count = 0
        self.episode_count = 0
        self.score_window = deque([0], maxlen=100)
        self.score1 = 0
        self.score2 = 0
        self.current_episode_length = 0
        self.xdata, self.ydata = [], []
        self.line, self.ax = None, None

    def on_step(self, reward1, reward2, done):
        self.step_count += 1
        self.current_episode_length += 1
        self.score1 += reward1
        self.score2 += reward2
        if done:
            self.episode_count += 1
            # append the maximum between the 2 agents scores
            self.score_window.append(max(self.score1, self.score2))
            self.score1 = 0
            self.score2 = 0
            self.current_episode_length = 0
            if self.ax is not None:
                self.ax.set_xlim(0, self.episode_count)
                self.xdata.append(self.episode_count)
                self.ydata.append(self.running_score())
                self.line.set_data(self.xdata, self.ydata)
                plt.pause(0.001)

    def running_score(self):
        return np.mean(self.score_window)

    def plot(self):
        fig, self.ax = plt.subplots()
        self.line, = plt.plot([], [], 'b-')

        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 50)
        self.ax.set_title('Multiagent Training')
        self.ax.set_xlabel('Episodes')
        self.ax.set_ylabel('Score')
        plt.pause(0.001)
