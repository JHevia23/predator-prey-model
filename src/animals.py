import numpy as np
import custom_utils as utils

class Wolf():

    COUNTER = 1

    def __init__(self, grid) -> None:
        '''
        // TODO: add stamina and hunger to simulation
        '''
        self.STAMINA = 100
        self.HUNGER = 0
        self.SPACE = grid
        self.STATE = utils.STATE_CONSTANT_ALIVE

        ## define simulation key parameters
        self.SMELL_RADIUS = 5
        self.SIGHT_RADIUS = utils.SIGHT_RADIUS_WOLF
        self.MOVE_STEPS = 2

        ## it's a tuple representing the coordinates for the wolf position
        initx = np.random.uniform(low=0, high=self.SPACE.XMAX)
        inity = np.random.uniform(low=0, high=self.SPACE.YMAX)
        self.pos = np.array((initx, inity))
        ## directional vector to simulate actual visible environment
        # its a unitary vector
        self.SIGHT_DIRECTION = utils.normalize(self.pos)

        ## define animal's type
        self.TYPE = utils.TYPE_CONSTANT_WOLF
        self.ID = utils.TYPE_CONSTANT_WOLF+str(self.COUNTER)
        grid.add_agent(self, agent_id=self.ID)
        Wolf.COUNTER+=1
        
    def move(self):
        '''
        The wolf smells for nearby sheep and chooses to move in the most efficient direction.
        If there are no sheep in the surroundings, it wanders randomly, moving 1 step into any direction
        '''
        ## check if there's a neighbouring sheep
        prey = self.locate_sheep(sense='SIGHT')
        
        if prey:
            print(f"PREY LOCATED {prey}")
            ## call the pos attribute of the Sheep Agent, and the next movement will be equal to the distance between
            # prey and predator
            next_pos = self.pos - prey.pos
            
        else:
            ## if there aren't, smell for sheep and move
            located_sheep = self.locate_sheep(sense='SMELL')
            if located_sheep:
                next_pos = ((self.pos - located_sheep.pos) / np.linalg.norm((self.pos - located_sheep.pos))) * self.MOVE_STEPS                
            else:
                ## add random noise to actual position to simulate wandering
                # next_pos = self.pos + np.random.uniform(0, 1, size=2)
                next_pos = np.random.uniform(-self.MOVE_STEPS, self.MOVE_STEPS, size=2)

        next_pos = utils.check_spatial_coherence(self, next_pos)
        
        ## assign new position to current position
        self.pos = self.pos + next_pos

        return self.pos

    def locate_sheep(self, sense='SIGHT'):
        '''
        The wolf smells around its Smell Radius and identifies presence of neighbouring
        sheep.

        The method scans the whole sheep population for calculating distances and then checks
        if any distance is less than the wolf's smell radius.

        - Return:
            - closest Sheep Agent or None if there aren't
        '''
        # identify sheeps that are visible to the wolf
        sheeps = np.array([sheep for sheep in self.SPACE.agent_population.values() if (sheep.TYPE == utils.TYPE_CONSTANT_SHEEP) &
                                                                                      (0 <= np.dot(sheep.SIGHT_DIRECTION, self.SIGHT_DIRECTION) <= 1)]) 
                                                                                    # only sheeps faced by the wolf

        distances = np.array([np.linalg.norm(self.pos - sheep.pos) for sheep in sheeps])
        min_distance = np.min(distances)
        
        if sense == 'SMELL':
            if min_distance <= self.SMELL_RADIUS:
                close_sheep = np.ravel(sheeps[np.argwhere(distances == min_distance)])

                if len(close_sheep) > 1:
                    ## if there are 2 sheeps with equally close distance, randomly pick one
                    close_sheep = np.random.choice(close_sheep)
            
                ## returns the closest Sheep Agent or None if there aren't
                return close_sheep[0]
            else:
                return None

        elif sense == 'SIGHT':

            if min_distance <= self.SIGHT_RADIUS:
                close_sheep = np.ravel(sheeps[np.argwhere(distances == min_distance)])

                if len(close_sheep) > 1:
                    ## if there are 2 sheeps with equally close distance, randomly pick one
                    
                    close_sheep = np.random.choice(close_sheep)
                
                print(f"CLOSE SHEEP DETECTED: {close_sheep[0]}")
                print("---" * 10)
                return close_sheep[0] ## return actual agent element
            else:
                ## return None if now sheeps were found
                return None
            
            


class Sheep():
    COUNTER = 1
    def __init__(self, grid) -> None:
        '''
        // TODO: add stamina and hunger to simulation
        '''
        self.STAMINA = 100
        self.HUNGER = 0
        self.SPACE = grid

        ## add simulation important parameters
        self.SIGHT_RADIUS = utils.SIGHT_RADIUS_SHEEP
        self.MOVE_STEPS = 3
        self.STATE = utils.STATE_CONSTANT_ALIVE

        initx = np.random.uniform(low=0, high=self.SPACE.XMAX)
        inity = np.random.uniform(low=0, high=self.SPACE.YMAX)

        ## it's a tuple representing the coordinates for the wolf position
        self.pos = np.array((initx, inity))
        ## directional vector to simulate actual visible environment
        # its a unitary vector
        self.SIGHT_DIRECTION = self.pos / np.linalg.norm(self.pos)

        ## add animal's type id
        self.TYPE = utils.TYPE_CONSTANT_SHEEP
        self.ID = utils.TYPE_CONSTANT_SHEEP+str(self.COUNTER)
        grid.add_agent(self, agent_id=self.ID)
        Sheep.COUNTER+=1


    def move(self):
        '''
        The movement decision function of sheeps is simpler: they will run from wolves if they're
        inside their sight radius.

        If the sheeps don't see a wolf, they wander around randomly. 
        
        // TODO: add grass to space
        '''
        # locate wolves that are visible to the sheep
        wolves = np.array([agent for agent in self.SPACE.agent_population.values() if (agent.TYPE == utils.TYPE_CONSTANT_WOLF) & 
                                                        (0 <= np.dot(agent.SIGHT_DIRECTION,self.SIGHT_DIRECTION) <= 1 )])

        distances = np.array([np.linalg.norm(wolf.pos - self.pos) for wolf in wolves])
        min_distance = np.min(distances)
        nearest_wolf = np.ravel(wolves[np.argwhere(distances == min_distance)])[0] ## extract element from raveled list

        if min_distance <= self.SIGHT_RADIUS:
            ## if a wolf is too nearby, the sheep will move further away from it
            distance_to_wolf = np.array(self.pos - nearest_wolf.pos) / np.linalg.norm(self.pos - nearest_wolf.pos)
            
            # create rotation matrix to prevent moving exactly in the opposite direction
            distance_to_wolf = utils.random_rotation(distance_to_wolf)
            
            # next_pos = self.pos + distance_to_wolf * self.MOVE_STEPS
            # next_pos = self.pos + distance_to_wolf
            next_pos = distance_to_wolf

        else:
            # next_pos = self.pos + np.random.uniform(0, 3, size=2)
            next_pos = np.random.uniform(-self.MOVE_STEPS, self.MOVE_STEPS, size=2)

        next_pos = utils.check_spatial_coherence(self, next_pos)
        
        ## update new position
        self.pos = self.pos + next_pos
        
        return self.pos