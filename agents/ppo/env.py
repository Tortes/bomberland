import os
import json
import math
import gym
import time
import websockets
import asyncio
import random

from math import pi
import numpy as np
from typing import Dict, List, Union
from server import Server
from state import State

ACTIONS = ["up", "down", "left", "right", "bomb", "detonate0", "detonate1", "detonate2"]

class Env():
    def __init__(self, server_port: str, agent: str):
        self.server = Server(server_port = server_port, agent = agent)
        self.last_state = None
        self.state = State(self)
        self.action_dim = len(ACTIONS)
        self.state_dim = 225
        self.action_space = ACTIONS

    # Define action
    def action_space(self):
        self.action_space_discrete = ["up", "down", "left", "right", "bomb", "detonate0", "detonate1", "detonate2"]
        self.action_dim = len(self.action_space_discrete)
        return gym.spaces.Discrete(self.action_dim)
    
    # Define observation
    def observation_space(self):
        self.state_dim = 225
        return gym.spaces.Discrete(self.state_dim)
    
    def get_reward(self, last_state, state):
        return 1
    
    def get_state(self):
        return self.state.update_state()
    
    def step(self, action: str):
        # Do action
        self.server.send_action(action)
        # Do obervation
        state = self.get_state()
        # Do reward
        reward = self.get_reward(self.last_state, state)
        self.last_state = state

        done = self.server.get_done()
        tick = self.server.get_tick()
        return np.asarray(state), reward, done, tick
    