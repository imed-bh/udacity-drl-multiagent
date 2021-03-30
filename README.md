# Udacity Deep Reinforcement Learning NanoDegree
## Project Collaboration and Competition

### Introduction
This project aims to train two agents which control rackets to bounce a ball over a net and keep it in play as much as possible.
![tennis](tennis.png)
*Unit ML-Agents Tennis Environment*

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

2. Place the file under this repository, and unzip (or decompress) the file. 
3. The code is written in Python 3 and uses the PyTorch library. You can install the requirements with:

```
conda create -n drlnd python=3.6
conda activate drlnd
pip install torch
pip install unityagents
```


### Environment Details

In this environment, two agents control rackets to bounce a ball over a net.
If an agent hits the ball over the net, it receives a reward of +0.1.
If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.
Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket.
Each agent receives its own, local observation.
Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, the agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,
* After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
* This yields a single **score** for each episode.
The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.


### Solving with (TODO)

In this section, we try to solve the problem using (TODO).
Details of the model are provided in `Report.md`. 

Note: you can skip the training part and use the provided pretrained `model.pt` to rollout a trained agent.

#### Training

To launch training, you need to run the train script with the file path to Unity environment as argument:

`python train.py <path_to_unity_env_here>`

For example on Linux: `python train.py Tennis_Linux/Tennis.x86_64`

You can customize the different options for training by modifying the variables at the top of file `train.py`.

### Rollout

To launch rollout, you need to run the rollout script with the file path to Unity environment as argument:

`python rollout.py <path_to_unity_env_here>`

For example on Linux: `python rollout.py Tennis_Linux/Tennis.x86_64`

You can customize the different options for rollout by modifying the variables at the top of file `rollout.py`.
