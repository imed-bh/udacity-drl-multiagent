import torch
import torch.nn.functional as F
import numpy as np

from src.metrics import Metrics
from src.model import Actor, Critic
from src.replay import ReplayBuffer, Experience

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DDPGConfig:
    def __init__(self,
                 buffer_size=100000,
                 batch_size=1024,
                 learning_rate=0.0001,
                 tau=0.001,
                 gamma=0.99,
                 fc1_units=256,
                 fc2_units=128):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma
        self.fc1_units = fc1_units
        self.fc2_units = fc2_units


class DDPGAgent:
    def __init__(self, env, config: DDPGConfig):
        self.env = env
        self.config = config
        self.replay_buffer = ReplayBuffer(config.buffer_size, config.batch_size)

        # Actor
        self.actor_current = Actor(3*env.state_size, env.action_size, config.fc1_units, config.fc2_units).to(device)
        self.actor_target = Actor(3*env.state_size, env.action_size, config.fc1_units, config.fc2_units).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor_current.parameters(), lr=config.learning_rate)

        # Critic
        self.critic_current = Critic(3*env.state_size, env.action_size, config.fc1_units, config.fc2_units).to(device)
        self.critic_target = Critic(3*env.state_size, env.action_size, config.fc1_units, config.fc2_units).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic_current.parameters(), lr=config.learning_rate)

        self.metrics = Metrics()

    def restore(self, actor_file, critic_file):
        self.actor_current.load_state_dict(torch.load(actor_file))
        self.critic_current.load_state_dict(torch.load(critic_file))

    def compute_action(self, state, epsilon=0):
        action = self.actor_current.action_values_for(state)
        if np.random.random() < epsilon:
            action += np.random.randn(self.env.action_size) * 0.2
            action = np.clip(action, -1, 1)
        return action

    def train(self, n_steps, update_every, print_every, epsilon_init=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        epsilon = epsilon_init
        states = self._warmup(epsilon)
        self.metrics.plot()

        for t_step in range(1, n_steps + 1):
            states = self._step(states, epsilon)
            epsilon = max(epsilon_min, epsilon * epsilon_decay)

            if t_step % update_every == 0:
                self._batch_train()
                if self._check_solved():
                    break

            if t_step % print_every == 0:
                print(f"Step #{t_step}" +
                      f", Running score {self.metrics.running_score():.2f}" +
                      f", Total episodes {self.metrics.episode_count}")

    def _warmup(self, epsilon):
        states = self.env.reset(train_mode=True)
        needed_experiences = max(0, self.replay_buffer.batch_size - len(self.replay_buffer))
        for i in range(needed_experiences):
            states = self._step(states, epsilon)
        return states

    def _step(self, states, epsilon):
        action1 = self.compute_action(states[0], epsilon)
        action2 = self.compute_action(states[1], epsilon)
        next_states, rewards, dones = self.env.step([action1, action2])
        done = any(dones)
        self.replay_buffer.add(Experience(states[0], action1, rewards[0], next_states[0], done))
        self.replay_buffer.add(Experience(states[1], action2, rewards[1], next_states[1], done))
        self.metrics.on_step(rewards[0], rewards[1], done)
        if done:
            return self.env.reset(train_mode=True)
        return next_states

    def _batch_train(self):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample()

        # Update Critic

        target_actions_next = self.actor_target(next_states)
        target_values_next = self.critic_target(next_states, target_actions_next).detach().max(1)[0].unsqueeze(1)
        target_values = rewards + (self.config.gamma * target_values_next * (1 - dones))
        expected_values = self.critic_current(states, actions)

        critic_loss = F.mse_loss(expected_values, target_values)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        for _ in range(10):
            self.critic_optimizer.step()
        self.critic_target.soft_update(self.critic_current, self.config.tau)

        # Update Actor

        current_actions = self.actor_current(states)
        actor_loss = -self.critic_current(states, current_actions).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        for _ in range(10):
            self.actor_optimizer.step()
        self.actor_target.soft_update(self.actor_current, self.config.tau)

    def _check_solved(self):
        if self.metrics.running_score() >= 0.5:
            print(f"\nEnvironment solved in {self.metrics.episode_count} episodes!\t" +
                  f"Average Score: {self.metrics.running_score():.2f}")
            torch.save(self.actor_current.state_dict(), "actor_model.pt")
            torch.save(self.critic_current.state_dict(), "critic_model.pt")
            return True

        return False
