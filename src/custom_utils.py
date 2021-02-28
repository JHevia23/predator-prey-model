TYPE_CONSTANT_WOLF = 'WOLF'
TYPE_CONSTANT_SHEEP = 'SHEEP'

import numpy as np

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
