#%%
from animals import Wolf, Sheep
from space import Grid
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import random
from celluloid import Camera

np.random.seed(1234)

grid = Grid(100, 100)

wolf1 = Wolf(grid)
wolf2 = Wolf(grid)

for i in range(20):
    new_sheep = Sheep(grid)

print(grid.agent_population.keys())


# this represents a single model step
for agent in grid.agent_population.values():
    grid.agent_population[agent.ID].pos = agent.move()


## let's try to animate this shit
fig = plt.figure(figsize=(10,10))
camera = Camera(fig)

for i in range(100):

    ## shuffle dict keys to prevent biasing behaviours from initialization
    grid_keys = list(grid.agent_population.keys())
    random.shuffle(grid_keys)

    for key in grid_keys:
        agent = grid.agent_population[key]
        grid.agent_population[agent.ID].pos = agent.move()

    xs = [agent.pos[0] for agent in grid.agent_population.values()]
    ys = [agent.pos[1] for agent in grid.agent_population.values()]
    c = ['red' if agent.TYPE == 'WOLF' else 'green' for agent in grid.agent_population.values()]

    plt.scatter(xs, ys, c=c)
    camera.snap()

animation = camera.animate(interval=20)

import matplotlib.animation as mp_anim
# Writer = mp_anim.PillowWriter(fps=30)
# writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

writer = mp_anim.PillowWriter(fps=100)
animation.save('test.gif', writer=writer)
# %%
