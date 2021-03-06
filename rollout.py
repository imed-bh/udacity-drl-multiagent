import sys
from pathlib import Path

from src.tennis_env import TennisEnv
from src.ddpg_agent import DDPGAgent, DDPGConfig


def get_env_path():
    if len(sys.argv) != 2:
        print("ERROR: invalid arguments list")
        print("Usage: rollout.py <path_to_unity_env>")
        sys.exit(1)
    return sys.argv[1]


if __name__ == "__main__":
    env = TennisEnv(get_env_path())
    agents = []
    for _ in range(2):
        agent = DDPGAgent(env, DDPGConfig(
            fc1_units=256,
            fc2_units=128,
        ))
        if Path("actor_model.pt").exists() and Path("critic_model.pt").exists():
            agent.restore("actor_model.pt", "critic_model.pt")
        agents.append(agent)
    env.run_episode(agents[0], agents[1])
    env.close()
