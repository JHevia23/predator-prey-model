import numpy as np
import custom_utils as utils
import matplotlib.pyplot as plt
from animals import Wolf, Sheep

class Grid():

    def __init__(self, len_x, len_y) -> None:
        
        ## override gym.Env parameters
        self.done = False
        # a 4dim vector for observations, because there are 4 directions
        self.observation = np.zeros([4, 3])


        self.XMAX = len_x
        self.YMAX = len_y
        # self.LAST_KEY = 0
        self.agent_population = {}
        self.dead_sheep_locations = []

    # def reset(self):
    #     '''
    #     reset method used for Reinforcement Learning implementation
    #     '''
    #     self.done = False
    #     self.observation = np.zeros([4,3])

    # def step(self, action):
    #     '''
    #     step method for reinforcement learning process

    #     //TODO: verify wolf ID is correctly input
    #     '''
    #     # 1. Update the environment state based on the action chosen
    #     ### manually move the wolf, this will change later
    #     if action == 0:
    #         self.agent_population['WOLF1'].pos = self.agent_population['WOLF1'].pos + np.array([self.agent_population['WOLF1'].MOVE_STEPS, 0])
    #     elif action == 90:
    #         self.agent_population['WOLF1'].pos = self.agent_population['WOLF1'].pos + np.array([0, self.agent_population['WOLF1'].MOVE_STEPS])
    #     elif action == 180:
    #         self.agent_population['WOLF1'].pos = self.agent_population['WOLF1'].pos + np.array([-self.agent_population['WOLF1'].MOVE_STEPS, 0])
    #     elif action == 270:
    #         self.agent_population['WOLF1'].pos = self.agent_population['WOLF1'].pos + np.array([0, -self.agent_population['WOLF1'].MOVE_STEPS])

    #     #2. Calculate the "reward" for the new state
    #     ### calculate dead sheep
    #     reward = self.update_population()

    #     #3. Update the states observation
    #     self.observation = self.get_observation()

    #     #4. Check if the game is over
    #     ### if there are no more sheep end the game
    #     '''
    #     //TODO: later we could end the game if the wolf dies of hunger or stamina runs out
    #     '''
    #     if len([s for s in self.agent_population.values() if s.TYPE == utils.TYPE_CONSTANT_SHEEP]) == 0:
    #         self.done = True

    #     return self.observation, reward, self.done


    # def get_observation(self):
    #     # directions to query
    #     directions = [0, 90, 180, 270]
    #     observation = np.zeros([4,3])
    #     # dic to save the features of each direction

    #     wolf = self.agent_population['WOLF1']

    #     for i, direction in enumerate(directions):
    #         sheeps_dic = self.perceive(direction)

    #         ## calculate f1 : direction feature
    #         sight_sheeps = sheeps_dic['SIGHT']
    #         vector_dif = [np.dot(sheep.SIGHT_DIRECTION, wolf.SIGHT_DIRECTION) for sheep in sight_sheeps]
    #         f1 = np.mean(vector_dif)
        
    #         ## calculate f2 : distance feature
    #         distances = [np.linalg.norm(sheep.pos - wolf.pos) for sheep in sight_sheeps]
    #         f2 = np.mean(distances)

    #         ## calculate f3 : opportunity gain (number of sheeps) of moving towards that direction, based on smell.
    #         smell_sheeps = np.array(list(sheeps_dic.values())).flatten()
    #         f3 = len(smell_sheeps)

    #         # update each observation
    #         observation[i] = np.array([f1, f2, f3])
        
    #     return observation
    
    # def perceive(self, direction):
    #     '''
    #     direction: angle to which the wolf is looking (0, 90, 180, 270)
    #     build a dictionary of the agents within smell radius and sight radius.

    #     copied from the Wolf class and adapted to fit the env

    #     - Returns
    #         - res_dict, keys: SIGHT, SMELL, values: list with agents
    #     '''
    #     res_dic = {}

    #     sheeps = [agent for agent in self.agent_population.values() if agent.TYPE == utils.TYPE_CONSTANT_SHEEP]
    #     wolf = self.agent_population['WOLF1']
    #     # set angle limits to perceive
    #     alpha_min = np.deg2rad(direction - 45)
    #     alpha_max = np.deg2rad(direction + 45)

    #     ## detect with SIGHT
    #     sight_sheep = [
    #         sheep for sheep in sheeps if (np.linalg.norm(sheep.pos - wolf.pos) <= wolf.SIGHT_RADIUS) &
    #                                      (alpha_min <= np.dot(sheep.SIGHT_DIRECTION, wolf.SIGHT_DIRECTION) <= alpha_max)
    #     ]

    #     res_dic['SIGHT'] = sight_sheep

    #     ## detect with SMELL
    #     smell_sheep = [
    #         sheep for sheep in sheeps if (np.linalg.norm(sheep.pos - wolf.pos) <= wolf.SMELL_RADIUS) &
    #                                      (alpha_min <= np.dot(sheep.SIGHT_DIRECTION, wolf.SIGHT_DIRECTION) <= alpha_max)
    #     ]

    #     res_dic['SMELL'] = smell_sheep

    #     return res_dic

    def add_agent(self, agent, agent_id):
        ## add agent to the space's population
        
        self.agent_population[agent_id] = agent
        self.population_ids = list(self.agent_population.keys())
    
    def get_population(self, type=utils.TYPE_CONSTANT_SHEEP):

        subset_population = []
        for agent in self.agent_population:
            if agent.TYPE == type:
                subset_population.append(agent)
        
        return subset_population

    def update_population(self):
        '''
        After each iteration, the grid will look up for 'DEAD' sheep and 
        remove them from the population
        '''
        # loop through sheeps
        sheeps = [agent for agent in self.agent_population.values() if agent.TYPE == utils.TYPE_CONSTANT_SHEEP]
        dead_sheep = 0

        for sheep in sheeps:
            if self.check_predators(sheep):
                # remove the sheep from the population
                self.agent_population.pop(sheep.ID)
                print(f"SHEEP {sheep.ID} DIED")
                self.dead_sheep_locations.append(sheep.pos)
                print(f"DEAD SHEEP COUNT {len(self.dead_sheep_locations)}")
                dead_sheep+=1
        return dead_sheep

    def check_predators(self, sheep):
        '''
        Returns True if there are predators at a fatal distance

        //TODO: recall that wolves = ... is commented out because now we have only one wolf (agent)
        //TODO: same for the distance calculation, which is no longer a vector but a scalar (distance between 2 agents)
        '''
        # wolves = [agent for agent in self.agent_population.values() if agent.TYPE == utils.TYPE_CONSTANT_WOLF]
        wolf = self.agent_population['WOLF1']
        distance = np.linalg.norm(wolf.pos - sheep.pos)
        # closest_distance = np.min(distances_id)

        if distance <= utils.SIGHT_RADIUS_WOLF:
            return True
        else:
            return False
        