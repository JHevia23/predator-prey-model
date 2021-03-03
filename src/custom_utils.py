TYPE_CONSTANT_WOLF = 'WOLF'
TYPE_CONSTANT_SHEEP = 'SHEEP'
STATE_CONSTANT_ALIVE = 'ALIVE'
STATE_CONSTANT_DEAD = 'DEAD'

SIGHT_RADIUS_WOLF = 5
SIGHT_RADIUS_SHEEP = 3


import numpy as np
import matplotlib.pyplot as plt
from celluloid import Camera
import random
import matplotlib.animation as mp_anim

## SIMULATION UTILS

def normalize(v):
    return v / np.linalg.norm(v)

def random_rotation(v):
    ## get random angle that ensure moving "forward", for sheep to escape
    angle = np.random.randint(-60, 60)
    c, s = np.cos(angle), np.sin(angle)
    ## build 2D rotation matrix
    R = np.array(((c, -s), (s, c)))
    v = np.dot(R, v)
    return v

def check_spatial_coherence(agent, next_pos):
    ## Check if the next position falls within the Grid's space
    if ((agent.pos[0] + next_pos[0]) >= agent.SPACE.XMAX) | ((agent.pos[0] + next_pos[0]) < 0):
        # if exceeded in the X axis, make the animal bounce back
        next_pos[0] = next_pos[0] * -1
    # repeat for y axis
    if ((agent.pos[1] + next_pos[1]) >= agent.SPACE.YMAX) | ((agent.pos[1] + next_pos[1]) < 0):
        next_pos[1] = next_pos[1] * -1

    return next_pos

## VISUALIZATION UTILS

def simulate(grid, n_iter=300, output_path='test.gif'):
    ## let's try to animate this shit
    fig = plt.figure(figsize=(10,10))
    camera = Camera(fig)

    for i in range(n_iter):

        ## shuffle dict keys to prevent biasing behaviours from initialization
        grid_keys = list(grid.agent_population.keys())
        random.shuffle(grid_keys)

        for key in grid_keys:
            ## instantiate each agent
            agent = grid.agent_population[key]

            ## call each agent's move method to simulate decisions, save new decided position in the grid's agent 
            grid.agent_population[agent.ID].pos = agent.move()
        
        grid.update_population()

        xs = [agent.pos[0] for agent in grid.agent_population.values()]
        ys = [agent.pos[1] for agent in grid.agent_population.values()]
        c = ['red' if agent.TYPE == 'WOLF' else 'green' for agent in grid.agent_population.values()]
        s = [100 if agent.TYPE == 'WOLF' else 50 for agent in grid.agent_population.values()]

        ## also plot dead body locations
        xs_dead = [loc[0] for loc in grid.dead_sheep_locations]
        ys_dead = [loc[1] for loc in grid.dead_sheep_locations]
        
        plt.scatter(xs, ys, c=c, s=s)
        plt.scatter(xs_dead, ys_dead, c='blue', s=75, marker='x')
        camera.snap()

    animation = camera.animate(interval=20)

    writer = mp_anim.PillowWriter(fps=100)
    animation.save(output_path, writer=writer)