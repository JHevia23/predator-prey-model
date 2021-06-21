from DQN_model import Wolf, Sheep, DQNAgent
from DQN_env import Farm
import numpy as np
import matplotlib.pyplot as plt



loss = []

action_space = 4
state_space = 12
max_steps = 1000
episode = 100

agent = DQNAgent(action_space, state_space)
farm = Farm(n_sheep=20)


for e in range(episode):
    state = env.reset()
    state = np.reshape(state, (1, state_space))
    score = 0
    for i in range(max_steps):
        action = agent.act(state)
        reward, next_state, done = farm.step(action)
        score += reward
        next_state = np.reshape(next_state, (1, state_space))
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        agent.replay()
        if done:
            print("episode: {}/{}, score: {}".format(e, episode, score))
            break
    loss.append(score)