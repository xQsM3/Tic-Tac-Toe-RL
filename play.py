from ttt import Client
from stable_baselines3 import A2C
from stable_baselines3 import PPO
from ttt import Board
import os

load_path = os.path.join("Training","Saved_Models")

env = Board()

agent = PPO.load(load_path,env)
client = Client(agent.policy)
print("sdhfj")
client.play_ttt()