import numpy as np
from animals import Sheep, Wolf
import custom_utils as utils
from collections import defaultdict


class Farm:

    def __init__(self, n_sheep, len_x=100, len_y=100):
        self.n_sheep = n_sheep
        self.len_x = len_x
        self.len_y = len_y

    def reset(self):
        sheep_id = 1

        # refresh population with n_sheep sheep
        self.agent_population = defaultdict(dict)
        for i in n_sheep:
            self.agent_population['SHEEP'][sheep_id] = Sheep(
                self.len_x, self.len_y, sheep_id)
            sheep_id = + 1

        # add wolf in random position
        self.agent_population['WOLF'][1] = Wolf(self.len_x, self.len_y, 1)

    # BUILD REWARD
    def update_population(self):
        '''
        After each iteration, the grid will look up for 'DEAD' sheep and 
        remove them from the population
        '''
        # loop through sheeps
        sheeps = [agent for agent in self.agent_population['SHEEP'].values()]

        dead_sheep = 0
        vulnerable_sheep = 0

        for sheep in sheeps:
            is_dead = self.check_predators(sheep)[0]
            distance = self.check_predators(sheep)[1]

            if is_dead:
                # remove the sheep from the population
                self.agent_population.pop(sheep.ID)
                print(f"SHEEP {sheep.ID} DIED")
                self.dead_sheep_locations.append(sheep.pos)
                print(f"DEAD SHEEP COUNT {len(self.dead_sheep_locations)}")
                dead_sheep += 1

            elif (np.dot(sheep.SIGHT_DIRECTION, self.agent_population['WOLF'][1].SIGHT_DIRECTION) < 1)
            & (distance < utils.SMELL_RADIUS_WOLF):
                vulnerable_sheep += 1

        reward = dead_sheep + 0.5*vulnerable_sheep

        return reward

    def check_predators(self, sheep):
        '''
        Returns True if there are predators at a fatal distance
        '''
        wolf = self.agent_population['WOLF1']
        distance = np.linalg.norm(wolf.pos - sheep.pos)

        if distance <= utils.SIGHT_RADIUS_WOLF:
            return True, distance
        else:
            return False, distance

    def step(self, action):
        
        ## move wolf
        self.agent_population['WOLF'][1].move(action)

        ## move sheep
        for sheep in self.agent_population['SHEEP'].values():
            sheep.move()

        # reward based on predated sheep
        reward = self.update_population()
        
        # wolf perceives new environment state
        next_state = self.agent_population['WOLF'][1].get_env_state()

        # done if no sheep remain alive
        done = len(list(self.agent_population['SHEEP'].values())) == 0

        return reward, next_state, done