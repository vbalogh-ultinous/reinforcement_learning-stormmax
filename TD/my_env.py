import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding

N_EDGES = 8
ACTION_MAP = np.array([
    [0,1,1,1,0,0,1,0],
    [1,0,1,0,0,1,0,1],
    [1,1,0,1,0,1,1,1],
    [1,0,1,0,1,0,1,1],
    [0,0,0,1,0,1,1,1],
    [0,1,1,0,1,0,1,1],
    [1,0,1,1,1,1,0,0],
    [0,1,1,1,1,1,0,0],

])

POINT_MAP = np.array([
    [0,1,1,1,0],
    [1,0,1,1,0],
    [1,1,0,1,1],
    [1,1,1,0,1],
    [0,0,1,1,0],
])

EDGE2POINTS = {
    0: [2,4],
    1: [4,3],
    2: [2,3],
    3: [2,0],
    4: [0,1],
    5: [3,1],
    6: [2,1],
    7: [0,3],
}

class MyEnv(gym.Env):
    metadata = {'render.modes': []}

    def __init__(self):
        self.n = N_EDGES
        self.visited = set()
        self.nsteps = 0
        self.validSteps = ACTION_MAP
        # -1: not visited position, 0: current position, 1: visited position
        self.observation_space = spaces.Box(low=-1, high=1, shape=(N_EDGES, 1), dtype=np.int)
        self.action_space = spaces.Discrete(N_EDGES)# step onto any edge
        self.current_state = tuple([-1 for i in range(N_EDGES)])
        self.edge_history = []
        self.point_history = []

    def isValid(self, action):
        if 0 not in self.current_state: # no actual actions taken yet (it's the start of the game)
            return True
        previous_pos = self.current_state.index(0)
        current_pos = action
        # print('valid:', previous_pos, '-->',current_pos,self.validSteps[previous_pos, current_pos] == 1)
        validSteps = self.validSteps[previous_pos, current_pos] == 1
        if not validSteps:
            return False
        else:
            return self.isValidHistory(action)

    def getPoints(self, edge=None):
        if edge == None:
            first = self.edge_history[0]
            second = self.edge_history[1]
            dir1 = EDGE2POINTS[first][0]
            dir2 = EDGE2POINTS[first][1]
            if dir1 in EDGE2POINTS[second]:
                self.point_history.append(dir2)
                self.point_history.append(dir1)
                remaining_point = [p for p in EDGE2POINTS[second] if p  != dir1]
                self.point_history.append(remaining_point[0])
            elif dir2 in EDGE2POINTS[second]:
                self.point_history.append(dir1)
                self.point_history.append(dir2)
                remaining_point = [p for p in EDGE2POINTS[second] if p != dir2]
                self.point_history.append(remaining_point[0])
            else:
                assert False, "EDGE2POINTS error"
        else:
            points = EDGE2POINTS[edge]
            prev_point = self.point_history[-1]
            next_point = [p for p in points if p != prev_point]
            self.point_history.append(next_point[0])

    def isValidHistory(self, action):
        if len(self.point_history) < 3:
            return True

        points = EDGE2POINTS[action]
        prev_point = self.point_history[-1]

        return prev_point in points

    def isGameOver(self, action):
        # print(self.visited)
        visited = action in self.visited
        notValid = not self.isValid(action)
        notValidHistory = not self.isValidHistory(action)
        # print('gamover visited:', visited, 'notvalid:', notValid, 'or', visited or notValid)
        return (visited or notValid or notValidHistory)

    def isWin(self):
        return N_EDGES == len(self.visited)

    def step(self, action):
        # print('step action:', action)
        gameOver = self.isGameOver(action)
        win = self.isWin()
        if gameOver:
            reward = -10
        else:
            self.visited.add(action)
            reward = -1

        self.nsteps += 1
        # print('previous state: ', self.current_state)
        if 0 in self.current_state:
            previous_pos = self.current_state.index(0)
            # print('previous:', previous_pos)
            l = list(self.current_state)
            l[previous_pos] = 1
            self.current_state = tuple(l)
            print('valid:', previous_pos, '-->', action, not gameOver, self.edge_history, self.point_history)

        current_pos = action
        l = list(self.current_state)
        l[current_pos] = 0
        self.current_state = tuple(l)

        self.edge_history.append(current_pos)
        if not gameOver:
            if 2 == len(self.edge_history):
                self.getPoints()
            elif len(self.edge_history) > 2:
                self.getPoints(action)

        obs = self.current_state
        # print('current_state', self.current_state)
        done = (win or gameOver)
        if win:
            info = {'win': True, 'gameover':False}
        elif gameOver:
            info = {'win': False, 'gameover': True}
        else: # game continues
            info = {'win': False, 'gameover': False}
        # print('info:', info)

        return obs, reward, done, info


    def reset(self):
        observation_space = tuple([-1 for i in range(N_EDGES)])
        # sampled_edge = self.action_space.sample()
        # observation_space[sampled_edge] = 0
        self.visited = set()
        self.current_state = observation_space
        self.edge_history = []
        self.point_history = []
        return self.current_state

    def render(self, mode='human'):
        return

