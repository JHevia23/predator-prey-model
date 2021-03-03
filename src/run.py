#%%
from animals import Wolf, Sheep
from space import Grid
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import random
from celluloid import Camera
import custom_utils as utils

np.random.seed(1234)

grid = Grid(100, 100)

## initialize agents
for i in range(5):
    new_wolf = Wolf(grid)

for i in range(20):
    new_sheep = Sheep(grid)

## run simulation and get GIF saved
print("AGENTS before running")
print(len(grid.agent_population.values()))

utils.simulate(grid, output_path='test2.gif')

print("AGENTS after running")
print(len(grid.agent_population.values()))
