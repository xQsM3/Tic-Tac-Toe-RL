import ttt
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv,VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy


log_path = os.path.join("Training","Logs")
if not os.path.isdir(log_path):
    os.makedirs(log_path)

save_path =os.path.join("Training","Saved_Models")
#if not os.path.isdir(save_path):
#    os.makedirs(save_path)

class Trainer(ttt.Client):
    def __init__(self,agent=None):
        super().__init__()
        self.players = {"x": ttt.Player("x",self.rendim,agent),
                        "o": ttt.Player("o",self.rendim,agent)}

    def train(self):
        while True:
            self.board.reset()
            while not self.board.done:
                self.render()
                action = self.players[self.pointer].action(self.state, self.img)
                self.step(action, self.pointer)
                self._movepointer()
                if self.done:
                    self.reset()
            break

env = ttt.Board()
#env = DummyVecEnv([lambda: env])
#env = VecFrameStack(env,n_stack=4)
model = PPO('MlpPolicy',env,verbose=1,tensorboard_log=log_path,learning_rate=1e-4)
model.learn(total_timesteps=15_000_000)
model.save(save_path)

'''
episodes = 5
for episode in range(1,episodes+1):
    obs = env.reset()
    done = False
    score = 0

    while not done:
        env.render()
        action = env.action_space.sample()
        obs,reward,done,info = env.step(action)
        score += reward
'''