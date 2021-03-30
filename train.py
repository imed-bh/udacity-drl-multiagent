import sys

from src.ddpg_agent import DDPGConfig, DDPGAgent
from src.tennis_env import TennisEnv

BUFFER_SIZE = 100000
BATCH_SIZE = 256
LEARNING_RATE = 0.002
TAU = 0.001
GAMMA = 0.99
FC1_UNITS = 128
FC2_UNITS = 64
N_STEPS = 10000000
UPDATE_EVERY = 10
PRINT_EVERY = 1000
EPS_INIT = 1.0
EPS_DECAY = 0.9999
EPS_MIN = 0.001


def get_env_path():
    if len(sys.argv) != 2:
        print("ERROR: invalid arguments list")
        print("Usage: train.py <path_to_unity_env>")
        sys.exit(1)
    return sys.argv[1]


if __name__ == '__main__':
    env = TennisEnv(get_env_path())
    agent = DDPGAgent(env, DDPGConfig(
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        tau=TAU,
        gamma=GAMMA,
        fc1_units=FC1_UNITS,
        fc2_units=FC2_UNITS,
    ))
    agent.train(N_STEPS, UPDATE_EVERY, PRINT_EVERY, EPS_INIT, EPS_DECAY, EPS_MIN)
    input('Press key to continue...')
    env.close()
