import numpy as np
import custom_utils as utils
import matplotlib.pyplot as plt

class Grid():

    def __init__(self, len_x, len_y) -> None:
        
        self.XMAX = len_x
        self.YMAX = len_y
        # self.LAST_KEY = 0
        self.agent_population = {}

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

