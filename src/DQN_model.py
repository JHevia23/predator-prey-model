'''
RL agent will be a DQN network based on the paper in reference
To consider:
    - Agent and environment must be separated and should communicate
    - Agent will include a replay-memory that helps to smooth out behav. learning
    - A NN is trained on for a number of episodes, and each episode includes several time-steps. Thus:
        - Initialize the NN
        - For each episode, restart the environment at t=0 and run T time-steps
            - for each time step:
                - agent reads environment's state & agent acts
                - environment is updated (new state) & reward is calculated
                - agent remembers the (state, action, new state, reward, is_done), saving the set in its memory
                - state is updated to new state
                - agent "replays", which represents the actual learning from previous experience (this is the NN training phase):
                    - a minibatch is built by uniformly sampling from the agent's memory
                    - a target value is calculated as a function of (reward, DQN model predict on batch for next_states sampled)
                        [except from the last state])
                    - adjust target to target_full with new predict on batch
                    - fit model
'''