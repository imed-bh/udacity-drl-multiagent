from unityagents import UnityEnvironment


class TennisEnv:
    def __init__(self, env_path):
        self.env = UnityEnvironment(file_name=env_path)
        self.brain_name = self.env.brain_names[0]
        brain = self.env.brains[self.brain_name]
        self.action_size = brain.vector_action_space_size
        self.state_size = brain.vector_observation_space_size

    def reset(self, train_mode):
        env_info = self.env.reset(train_mode)[self.brain_name]
        return env_info.vector_observations[0]

    def step(self, action):
        env_info = self.env.step(action)[self.brain_name]
        return env_info.vector_observations[0], env_info.rewards[0], env_info.local_done[0]

    def close(self):
        self.env.close()

