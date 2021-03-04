import numpy as np
import custom_utils as utils
import matplotlib.pyplot as plt
from animals import Wolf, Sheep

class Grid():

    def __init__(self, len_x, len_y) -> None:
        
        self.XMAX = len_x
        self.YMAX = len_y
        # self.LAST_KEY = 0
        self.agent_population = {}
        self.dead_sheep_locations = []

        

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
        After each iteration, the grid will look up for 'DEAD' sheep and remove them from the population
        '''
        # loop through sheeps
        sheeps = [agent for agent in self.agent_population.values() if agent.TYPE == utils.TYPE_CONSTANT_SHEEP]
        for sheep in sheeps:
            if self.check_predators(sheep):
                # remove the sheep from the population
                self.agent_population.pop(sheep.ID)
                print(f"SHEEP {sheep.ID} DIED")
                self.dead_sheep_locations.append(sheep.pos)
                print(f"DEAD SHEEP COUNT {len(self.dead_sheep_locations)}")



    def check_predators(self, sheep):
        '''
        Returns True if there are predators at a fatal distance
        '''
        wolves = [agent for agent in self.agent_population.values() if agent.TYPE == utils.TYPE_CONSTANT_WOLF]
        distances_id = [np.linalg.norm(wolf.pos - sheep.pos) for wolf in wolves]
        closest_distance = np.min(distances_id)

        if closest_distance <= utils.SIGHT_RADIUS_WOLF:
            return True
        else:
            return False
        