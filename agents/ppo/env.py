import os
import json
import math
import gym
import websockets
import asyncio
import random

from math import pi
import numpy as np
from typing import Dict, List, Union

from game_state import GameState

actions = ["up", "down", "left", "right", "bomb", "detonate"]

class Env():
    def __init__(self, fwd_model_uri: str):
        self.connection_url = fwd_model_uri
        self._client = GameState(fwd_model_uri)

    def init():
        self._client.set_game_tick_callback(self._on_game_tick)
        self.set_next_state_callback(self._on_next_game_state)


    async def connect(self):
        self.connection = await websockets.connect(self._connection_string)
        if self.connection.open:
            return self.connection 
        
        loop = asyncio.get_event_loop()
        client_connection = loop.run_until_complete(self._client.connect())

        client_fwd_connection = loop.run_until_complete(
            self._client_fwd.connect())

        loop = asyncio.get_event_loop()
        loop.create_task(self._client._handle_messages(client_connection))
        loop.create_task(
            self._client_fwd._handle_messages(client_fwd_connection))
        loop.run_forever()

    async def _on_game_tick(self, tick_number, game_state):
        await self._send_eval_next_state()
        random_action = self.generate_random_action()
        if random_action in ["up", "left", "right", "down"]:
            await self._client.send_move(random_action)
        elif random_action == "bomb":
            await self._client.send_bomb()
        elif random_action == "detonate":
            bomb_coordinates = self._get_bomb_to_detonate(game_state)
            if bomb_coordinates != None:
                x, y = bomb_coordinates
                await self._client.send_detonate(x, y)
        else:
            print(f"Unhandled action: {random_action}")

    async def _on_next_game_state(self, state):
        # print(state)
        pass

    def generate_random_action(self):
        actions_length = len(actions)
        return actions[random.randint(0, actions_length - 1)]

    def set_next_state_callback(self, next_state_callback):
        self._next_state_callback = next_state_callback

    # Define action
    def action_space(self):
        self.action_space_discrete = ["up", "down", "left", "right", "bomb", "detonate0", "detonate1", "detonate2"]
        self.action_dim = len(self.action_space_discrete)
        return gym.spaces.Discrete(self.action_dim)
    
    # Define observation
    def observation_space(self):
        self.state_dim = 225
        return gym.spaces.Discrete(self.state_dim)