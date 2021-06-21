import numpy as np
from animals import Sheep
from collections import defaultdict

class Farm:
    
    def __init__(self, n_sheep, len_x=100, len_y=100):
        self.n_sheep = n_sheep
        self.len_x = len_x
        self.len_y = len_y

    def reset(self):
        sheep_id = 1

        ## refresh population with n_sheep sheep
        self.population = defaultdict(dict)
        for i in n_sheep:
            self.population['SHEEP'][sheep_id] = Sheep(self.len_x, self.len_y, sheep_id)
            sheep_id =+ 1