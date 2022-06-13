import gym
import torch
from gym import Env
from gym.spaces.discrete import Discrete
from gym.spaces.multi_discrete import MultiDiscrete
import cv2 as cv
import numpy as np
import stable_baselines3 as sb

class Board(Env):
    def __init__(self):
        super().__init__()
        # render stuff
        self.rendim = 300
        self.renthick = 3
        self.rencolor = (0,0,0)

        #environment stuff
        self.observation_space = MultiDiscrete([3]*9)
        self.action_space =MultiDiscrete([3]*2)
        self.reset()

    def reset(self):
        self.state = np.zeros((3, 3))
        self._drawEmptyBoard()
        self.done = False
        self.pointer = np.random.choice(["x","o"]) #random choose a player to start from
        self.stepcount = 0
        state = self._swap_ids() #swaps ids
        return state.flatten()

    def step(self,action):
        x,y = action
        id = self.pointer
        self.stepcount += 1

        # perform action if field is not blocked
        if self.state[y, x] == 0:
            self._drawActionOnBoard(action)
            self.state[y, x] = 2 if id == "x" else 1
            # get reward for the action
            reward, self.done, self.info = self._get_reward(action)
            # swap IDs x / o such that agent sees itself always as player X (while training against itself)
            state = self._swap_ids().flatten()
            # move ID pointer
            self._movepointer()
        # if field is blocked, punish the agent and dont perform action
        else:
            state = self.state.copy().flatten()
            reward = -15
        print(f"REWARD: {reward}")
        return state, reward, self.done, {}
    def render(self):
        cv.imshow("Tic Tac Toe",self.img)
        cv.waitKey(1)
    def _get_reward(self,action):
        id = self.pointer #id of current player
        id_int_oponent = 2 if id == "o" else 1 #id of oponent in integer
        state = self.state.copy() #create dummy state
        row,col = action



        # check if player won
        captured_col = np.all((state == 2), axis=0).any() or np.all((state == 1), axis=0).any()
        captured_row = np.all((state == 2), axis=1).any() or np.all((state == 1), axis=1).any()
        captured_diagonal_1 = ((state[0, 0] == state[1, 1] == state[2, 2]) and state[0, 0] != 0)
        captured_diagonal_2 = ((state[2, 0] == state[1, 1] == state[0, 2]) and state[2, 0] != 0)

        if captured_col or captured_row or captured_diagonal_1 or captured_diagonal_2:
            reward = 10
            return reward,True,[f"Player {id} won",id]
        # check if tie
        if not np.any((state) == 0):
            reward = 2
            return reward,True,["Tie",None]

        # check if player will loose next step
        lost_col,lost_row = False,False
        for row,col in zip(state,state.T):
            if (np.count_nonzero(row == id_int_oponent) == 2) and np.count_nonzero(row == 2) == 2:
                lost_row = True
                break
            if (np.count_nonzero(col == id_int_oponent) == 2) and np.count_nonzero(col == 2) == 2:
                lost_col = True
                break

        d1 = np.array([state[0,0],state[1,1],state[2,2]])
        d2 = np.array([state[2,0],state[1,1],state[0,2]])
        lost_d1 = (np.count_nonzero(d1 == id_int_oponent) == 2) and np.count_nonzero(d1 == 2) == 2
        lost_d2 = (np.count_nonzero(d2 == id_int_oponent) == 2) and np.count_nonzero(d2 == 2) == 2

        if lost_col or lost_row or lost_d1 or lost_d2:

            reward = -10
            return reward,False,["Loose possible", None]

        # punish with -1 for every move
        reward = -1

        return reward,False,"running"
    def _swap_ids(self):
        # if its player o turn: switch state IDs before passing to agent, such that agent always sees itself as player x
        if self.pointer == "o":
            state = self.state.copy()
            state[self.state==1] = 2
            state[self.state==2] = 1
            return state
        return self.state

    def _drawEmptyBoard(self):
        dim = self.rendim
        t = self.renthick
        img = np.zeros((dim+t+50,dim+t,t)) + 255 #init white board
        cv.line(img,(dim//3,0),(dim//3,dim),self.rencolor,3) # draw left colume line
        cv.line(img,(dim-dim//3,0),(dim-dim//3,dim),self.rencolor,3) # draw right column line
        cv.line(img,(0,dim-dim//3),(dim,dim-dim//3),self.rencolor,3) # draw right column line
        cv.line(img,(0,dim//3),(dim,dim//3),self.rencolor,3) # draw right column line
        self.img = img
    def _drawActionOnBoard(self,action):
        id = self.pointer
        # expand row / col of action to board dimensions
        row,col = action
        size = self.rendim//3 //2
        rowexpand = row * self.rendim//3 + self.rendim//3 - size
        colexpand = col * self.rendim//3 + self.rendim//3 - size
        size -= 10
        # draw an x if player id is x, otherwise draw an o
        if id =="x":
            cv.line(self.img,(rowexpand-size,colexpand-size),(rowexpand+size,colexpand+size),self.rencolor,self.renthick)
            cv.line(self.img,(rowexpand-size,colexpand+size),(rowexpand+size,colexpand-size),self.rencolor,self.renthick)
        else:
            cv.circle(self.img,(rowexpand,colexpand),size,self.rencolor,self.renthick)
    def _movepointer(self):
        self.pointer = "x" if self.pointer == "o" else "o"


class Player():

    def __init__(self,id,dim,agent=None):
        self.id = id
        self.dim = dim
        self.agent = agent
        self.mousepos = None
        cv.namedWindow("Tic Tac Toe")

    def action(self,obs,img):
        cv.setMouseCallback("Tic Tac Toe", self._callback)
        if self.agent != None: # is this player an AI ?
            obs = torch.FloatTensor(obs).flatten().unsqueeze(0) #prepare obsercation for model
            action = self.agent(obs)[0] #predict
            action = action.tolist()[0] # convert prediction to list
        else: # nope, its a human
            self.mousepos = None
            # wait for mouse callback and compute action from it
            while True:
                cv.imshow("Tic Tac Toe",img)
                cv.waitKey(100)

                if self.mousepos != None:
                    # transform mouse input into action space
                    x,y = self.mousepos[0]

                    if self.dim/3 > x:
                        x = 0
                    elif self.dim*2/3 > x:
                        x = 1
                    else:
                        x = 2

                    if self.dim/3 > y:
                        y = 0
                    elif self.dim*2/3 > y:
                        y = 1
                    else:
                        y = 2

                    #only break out of while loop if action is not blocked on Board yet
                    if obs[y,x] == 0:
                        break


            action = [x,y]

        return action

    def _callback(self,event,x,y,flags,param):
        if event == cv.EVENT_LBUTTONDOWN:
            self.mousepos = [(x,y)]


class Client(Board):
    def __init__(self,agent=None):
        super().__init__()
        self.players = {"x": Player("x",self.rendim,agent),
                        "o": Player("o",self.rendim)}
        #self.board = Board()
    def play_ttt(self):
        while True:
            self.reset()

            while not self.done:
                self.render()
                action = self.players[self.pointer].action(self.state, self.img)
                self.step(action)

                if self.stepcount >= 20:
                    break

            cancel = self.finishscreen()
            if cancel:
                break
    def finishscreen(self):
        f = self.img.copy()
        winnerid = self.info[1]
        if winnerid == None:
            finishtext = "Tie!"
        else:

            if self.players[winnerid].agent == None: #then its a human
                finishtext = "Human WON!"
            else:
                finishtext = "Agent WON!"
        cv.putText(f,finishtext,(50,350),cv.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2,cv.LINE_AA)
        cv.imshow("Tic Tac Toe",f)
        if cv.waitKey() & 0xFF == 27:
            return True
        else:
            return False


