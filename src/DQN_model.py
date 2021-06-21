# '''
# RL agent will be a DQN network based on the paper in reference
# To consider:
#     - Agent and environment must be separated and should communicate
#     - Agent will include a replay-memory that helps to smooth out behav. learning
#     - A NN is trained on for a number of episodes, and each episode includes several time-steps. Thus:
#         - Initialize the NN
#         - For each episode, restart the environment at t=0 and run T time-steps
#             - for each time step:
#                 - agent reads environment's state & agent acts
#                 - environment is updated (new state) & reward is calculated
#                 - agent remembers the (state, action, new state, reward, is_done), saving the set in its memory
#                 - state is updated to new state
#                 - agent "replays", which represents the actual learning from previous experience (this is the NN training phase):
#                     - a minibatch is built by uniformly sampling from the agent's memory
#                     - a target value is calculated as a function of (reward, DQN model predict on batch for next_states sampled)
#                         [except from the last state])
#                     - adjust target to target_full with new predict on batch
#                     - fit model
#                     '''

from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Sequential
import random
import numpy as np
from collections import defaultdict, deque


class Sheep:

    def __init__(self, xmax, ymax, sheep_id):
        '''
        Constructor for sheep, must be space-agnostic. Should work with any instance of environment.

        SHEEP PARAMS:
            - SIGHT RADIUS
            - MOVE_STEPS

        INDIVIDUAL SHEEP PARAMS:
            - POSITION (as a [x,y] randomized vector)
            - SIGHT DIRECTION (facing direction)
            - ID
        '''

        # self.STAMINA = 100
        # self.HUNGER = 0

        # add simulation important parameters
        self.SIGHT_RADIUS = utils.SIGHT_RADIUS_SHEEP
        self.MOVE_STEPS = utils.MOVE_STEPS_SHEEP
        self.STATE = utils.STATE_CONSTANT_ALIVE

        initx = np.random.uniform(low=0, high=xmax)
        inity = np.random.uniform(low=0, high=ymax)

        # it's a tuple representing the coordinates for the wolf position
        self.pos = np.array((initx, inity))
        # directional vector to simulate actual visible environment
        # its a unitary vector
        self.SIGHT_DIRECTION = self.pos / np.linalg.norm(self.pos)

        # add animal's type id
        self.TYPE = utils.TYPE_CONSTANT_SHEEP
        self.ID = utils.TYPE_CONSTANT_SHEEP+str(sheep_id)

    def move(self, farm):
        '''
        At each environment step, the sheep will be passed the farm where it is located as an arg.
        Thus, each sheep can SEE whether there's a nearby wolf or not and act on it.
        '''

        # locate wolves that are visible to the sheep
        # TODO: farm must be a dictionary with all agents present and their animal type
        # wolves = np.array([agent for agent in farm.agent_population['WOLF'].values() if (
        #     0 <= np.dot(agent.SIGHT_DIRECTION, self.SIGHT_DIRECTION) <= 1)])

        # # this line still holds for many-wolves scenario
        # distances = np.array([np.linalg.norm(wolf.pos - self.pos)
        #                      for wolf in wolves])
        # min_distance = np.min(distances)
        # nearest_wolf = np.ravel(wolves[np.argwhere(distances == min_distance)])[
        #     0]  # extract element from raveled list

        wolf = farm.agent_population['WOLF'][1]
        min_distance = np.linalg.norm(wolf.pos - self.pos)

        if min_distance <= self.SIGHT_RADIUS:
            # if a wolf is too nearby, the sheep will move further away from it
            distance_to_wolf = np.array(
                self.pos - nearest_wolf.pos) / np.linalg.norm(self.pos - nearest_wolf.pos)

            # create rotation matrix to prevent moving exactly in the opposite direction, add some randomness to movement
            distance_to_wolf = utils.random_rotation(distance_to_wolf)

            next_pos = distance_to_wolf

        else:
            next_pos = np.random.uniform(-self.MOVE_STEPS,
                                         self.MOVE_STEPS, size=2)

        # correction to protect against off-bounds movement
        next_pos = utils.check_spatial_coherence(self, next_pos)

        # update new position and direction
        self.pos = self.pos + next_pos
        self.SIGHT_DIRECTION = self.pos / np.linalg.norm(self.pos)

        return self.pos


class Wolf:

    def __init__(self, xmax, ymax, wolf_id):
        '''
        Constructor for wolf, must be space-agnostic. Should work with any instance of environment.

        SHEEP PARAMS:
            - SIGHT RADIUS
            - SMELL RADIUS
            - MOVE_STEPS

        INDIVIDUAL SHEEP PARAMS:
            - POSITION (as a [x,y] randomized vector)
            - SIGHT DIRECTION (facing direction)
            - ID
        '''

        # self.STAMINA = 100
        # self.HUNGER = 0
        self.SPACE = grid
        self.STATE = utils.STATE_CONSTANT_ALIVE

        # define simulation key parameters
        self.SMELL_RADIUS = utils.SMELL_RADIUS_WOLF
        self.SIGHT_RADIUS = utils.SIGHT_RADIUS_WOLF
        self.MOVE_STEPS = utils.MOVE_STEPS_WOLF
        self.DEATH_COUNT = 0

        # it's a tuple representing the coordinates for the wolf position
        initx = np.random.uniform(low=0, high=xmax)
        inity = np.random.uniform(low=0, high=ymax)
        self.pos = np.array((initx, inity))

        # directional vector to simulate actual visible environment
        # its a unitary vector
        self.SIGHT_DIRECTION = utils.normalize(self.pos)

        # define animal's type
        self.TYPE = utils.TYPE_CONSTANT_WOLF
        self.ID = utils.TYPE_CONSTANT_WOLF+str(wolf_id)

    def perceive(self, direction, farm):
        '''
        Build a dictionary of the agents within smell radius and sight radius.
        Used for Qvalues calculation.

        - params
            - direction: angle to which the wolf is looking (0, 90, 180, 270)

        - returns
            - res_dict with keys: [SIGHT, SMELL] and values: list with sheep agents
        '''
        res_dic = {}

        sheeps = list(farm.agent_population['SHEEP'].values())

        # set angle limits to perceive
        # 45 degrees because we have 4 directions possible
        alpha_min = np.deg2rad(direction - 45)
        alpha_max = np.deg2rad(direction + 45)

        # detect with SIGHT
        sight_sheep = [
            sheep for sheep in sheeps if (np.linalg.norm(sheep.pos - self.pos) <= self.SIGHT_RADIUS) &
                                         (alpha_min <= np.dot(
                                             sheep.SIGHT_DIRECTION, self.SIGHT_DIRECTION) <= alpha_max)
        ]

        res_dic['SIGHT'] = sight_sheep

        # detect with SMELL
        smell_sheep = [
            sheep for sheep in sheeps if (np.linalg.norm(sheep.pos - self.pos) <= self.SMELL_RADIUS) &
                                         (alpha_min <= np.dot(
                                             sheep.SIGHT_DIRECTION, self.SIGHT_DIRECTION) <= alpha_max)
        ]

        res_dic['SMELL'] = smell_sheep

        return res_dic

    def get_env_state(self):
        # directions to query
        directions = [0, 90, 180, 270]

        # dic to save the features of each direction
        features_dic = defaultdict(list)
        state = []

        for direction in directions:
            sheeps_dic = self.perceive(direction)

            # calculate f1 : direction feature
            sight_sheeps = sheeps_dic['SIGHT']
            vector_dif = [np.dot(sheep.SIGHT_DIRECTION, self.SIGHT_DIRECTION)
                          for sheep in sight_sheeps]
            features_dic[direction].append(np.mean(vector_dif))

            # calculate f2 : distance feature
            distances = [np.linalg.norm(sheep.pos - self.pos)
                         for sheep in sight_sheeps]
            features_dic[direction].append(np.mean(distances))

            # calculate f3 : opportunity gain (number of sheeps) of moving towards that direction, based on smell.
            smell_sheeps = np.array(
                list(sheeps_dic['SMELL'].values())).flatten()
            features_dic[direction].append(len(smell_sheeps))

            # build final states array, by adding each direction features
            state.extend(list(features_dic[direction].values()))

        return state

    def move(self, action):

        if action == 1:
            # 90 deg move
            self.pos = np.sum(self.pos, [0, self.MOVE_STEPS])
        elif action == 2:
            # 180 deg move
            self.pos = np.sum(self.pos, [-self.MOVE_STEPS, 0])
        elif action == 3:
            # 270 deg move
            self.pos = np.sum(self.pos, [0, -self.MOVE_STEPS])
        elif action == 4:
            self.pos = np.sum(self.pos, [self.MOVE_STEPS, 0])


class DQNAgent:

    def __init__(self, action_space, state_space):

        self.action_space = action_space  # 4 directions
        self.state_space = state_space  # 12 (3 feats per direction)
        self.epsilon = 1  # for exploration vs explotation policy

        # DL params
        self.gamma = .95
        self.batch_size = 64
        self.epsilon_min = .01
        self.epsilon_decay = .995
        self.learning_rate = 0.001
        self.memory = deque(maxlen=100000)

        self.model = self.build_model()

    def build_model(self):
        ''' 
        Instantiate NN architecture, optimizer and loss.
        '''

        model = Sequential()
        model.add(Dense(64, input_shape=(self.state_space,), activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_space, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        '''
        Save environment state to memory for further sampling for trainning
        '''
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        '''
        Choose between exploration or explotation approach and return action
        '''
        if np.random.rand() <= self.epsilon:
            # randrange always returns an random int from list(range(0, self.action_space))
            return random.randrange(self.action_space)

        act_values = self.model.predict(state)

        # TODO: act_values[0] is a list??
        return np.argmax(act_values[0])

    def replay(self):
        '''
        Access memory and perform learning with replay-approach.
        Sample observations from agent's memory to build a trainning epoch.
        '''

        if len(self.memory) < self.batch_size:
            # accumulate observations until enough memory to replay-train
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        # TODO: WHAT's this?????
        targets = rewards + self.gamma * \
            (np.amax(self.model.predict_on_batch(next_states), axis=1))*(1-dones)

        targets_full = self.model.predict_on_batch(states)

        ind = np.array([i for i in range(self.batch_size)])
        targets_full[[ind], [actions]] = targets

        self.model.fit(states, targets_full, epochs=1, verbose=0)

        # decrease epsilon to shift policy towards a more conservative one
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
