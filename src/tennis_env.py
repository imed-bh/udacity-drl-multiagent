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
        return env_info.vector_observations

    def step(self, action):
        env_info = self.env.step(action)[self.brain_name]
        return env_info.vector_observations, env_info.rewards, env_info.local_done

    def close(self):
        self.env.close()

    def run_episode(self, agent1, agent2, fast=False):
        states = self.reset(train_mode=fast)
        score1 = 0
        score2 = 0
        done = False
        while not done:
            action1 = agent1.compute_action(states[0], epsilon=0)
            action2 = agent2.compute_action(states[1], epsilon=0)
            states, rewards, dones = self.step([action1, action2])
            score1 += rewards[0]
            score2 += rewards[1]
            if (rewards[0] != 0 or rewards[1] != 0) and not fast:
                print(f"Scores: {score1:.2f}, {score2:.2f}")
        return max(score1, score2)


