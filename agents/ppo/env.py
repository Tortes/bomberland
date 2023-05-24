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
    def __init__(self):
        self.server = Server()
        self.state = State()
        self.last_state = None

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
        return
    
    def get_state(self):
        # get general variables
        state = self.server.get_state()
        my_unit_id = self.server.unit
        unit_state = state['unit_state']
        my_unit = unit_state[my_unit_id]
        my_agent_id = my_unit['agent_id']
        entities = state['entities']
        tick = state['tick']
        # sort team IDs so that my team comes first
        agent_order = ['a','b']
        agent_order.sort(key=lambda x: int(x != my_agent_id))
        agent_order = ['x'] + agent_order
        # create a dictionary of all units grouped by team. team 'x' is us
        agent2units = {
            'x' : my_unit_id,
            'a' : [u for u in ['c', 'e', 'g'] if u != my_unit_id],
            'b': [u for u in ['d', 'f', 'h'] if u != my_unit_id],
        }
        # we will now loop through the teams and units to create different grayscale images
        # where each image represents one aspect of the game state, such as HP or bombs
        layers = []
        for agent in agent_order:
            tmp = np.zeros([15,15,5], np.float32)
            for unit_id in agent2units[agent]:
                unit = unit_state[unit_id]
                cux, cuy = unit['coordinates']
                tmp[cuy, cux, 0] = 1.0
                tmp[cuy, cux, 1] = float(max(0,unit['hp']))
                tmp[cuy, cux, 2] = float(max(0,unit['invulnerability'] - tick)) / 6.0
                tmp[cuy, cux, 3] = min(float(unit['inventory']['bombs']), 7)
                tmp[cuy, cux, 4] = min(float(unit['blast_diameter']) / 3.0, 7)
            layers.append((f'agent {agent} positions', tmp[:,:,0], '01.0f'))
            layers.append((f'agent {agent} HP', tmp[:,:,1], '01.0f'))
            layers.append((f'agent {agent} invulnerability', tmp[:,:,2], '3.1f'))
            layers.append((f'agent {agent} bombs', tmp[:,:,3], '01.0f'))
            layers.append((f'agent {agent} blast_diameter', tmp[:,:,4], '01.0f'))

        # draw the environment HP and fire expiration times into a map
        tiles = np.zeros([15, 15], np.uint8)
        for e in entities:
            type = e['type']
            x, y = e['x'], e['y']
            if type in ['m', 'o', 'w']:
                tiles[y, x] = e.get('hp', 99)
            elif type in ['x']:
                tiles[y, x] = 100 + max(0, e.get('expires', tick + 99) - tick - 1)

        layers.append(('environment HP 1', np.float32(tiles == 1), '01.0f'))
        layers.append(('environment HP 2', np.float32(tiles == 2), '01.0f'))
        layers.append(('environment HP 3', np.float32(tiles == 3), '01.0f'))
        layers.append(('environment HP 99', np.float32(tiles == 99), '01.0f'))

        fire_time = np.maximum(np.float32(tiles) - 100, np.zeros_like(tiles)) / 100.0
        layers.append(('fire time', np.float32(fire_time), '3.1f'))

        # draw bomb, ammo, and powerup positions
        for type in ['b', 'a', 'bp']:
            layer = np.zeros([15, 15], np.float32)
            for e in entities:
                if e['type'] != type: continue
                layer[e['y'], e['x']] = 1.0
            layers.append((f'entity {type} pos', layer, '01.0f'))

        # how long will that bomb or fire still remain?
        for type in ['b', 'x']:
            layer = np.zeros([15, 15], np.float32)
            for e in entities:
                if e['type'] != type: continue
                layer[e['y'], e['x']] = float(e.get('expires',9999) > tick+1)
            layers.append((f'entity {type} remain', layer, '01.0f'))

        # how long until that bomb expires?
        for type in ['b']:
            layer = np.zeros([15, 15], np.float32)
            for e in entities:
                if e['type'] != type: continue
                if 'expires' not in e: continue
                layer[e['y'], e['x']] = float(e['expires'] - tick) / 40.0
            layers.append((f'entity {type} expires', layer, '3.1f'))

        # we need to specify where the game world ends because we will crop it to be relative to the unit
        layers.append(('world', np.ones([15, 15], np.float32), '01.0f'))

        # crop our observations to be relative to the unit
        cx, cy = unit_state[my_unit_id]['coordinates']
        view = 7
        sx, ex = max(0,cx-view), min(cx+view,15)+1
        sy, ey = max(0,cy-view), min(cy+view,15)+1
        layers = [(k,v[sy:ey,sx:ex],f) for (k,v,f) in layers]
        sx, ex = max(0,view-cx), max(0,cx-view)
        sy, ey = max(0,view-cy), max(0,cy-view)
        layers = [(k,np.pad(v,[(sy,ey),(sx,ex)]),f) for (k,v,f) in layers]
        return layers
    
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
    
class Server():
    def __init__(self):
        self.client = GameState()
        self.state = None
        self.agent = None
        self.unit = None
        self.hp = 0
        self.world_size = [15, 15]
        self.tick = 0
        self.bomb = {"detonate0" : {"x" : -1, "y" : -1}, 
                     "detonate1" : {"x" : -1, "y" : -1}, 
                     "detonate2" : {"x" : -1, "y" : -1}}
        
        self.client.set_game_tick_callback(self.tick_callback)

    async def send_action(self, action: str):
        if action in ["up", "left", "right", "down"]:
            await self.client.send_move(action, self.unit)
        elif action == "bomb":
            await self._client.send_bomb(self.unit)
        elif action in ["detonate0", "detonate1", "detonate2"]:
            bomb_coordinates = self.bomb[action]
            if bomb_coordinates.x >= 0 and bomb_coordinates.y > 0:
                await self._client.send_detonate(bomb_coordinates.x, 
                                                 bomb_coordinates.y, self.unit)
        else:
            print(f"Unhandled action: {action} for unit {self.unit}")

    async def tick_callback(self, tick_number, game_state):
        self.tick = tick_number
        self.state = game_state
        print(self.tick)
        print(self.state)

    def get_state(self):
        return self.state
    
    def get_done(self):
        return
    
    def get_tick(self):
        return self.tick
    
    async def update_bomb():
        return
    
class State():
    def __init__(self):
        return
    
    def load_state(state):
        for elem in state:
            print(elem)

async def connect_websocket(env):
    async with websockets.connect("ws://127.0.0.1:3000/?role=agent&agentId=agentA&name=defaultName") as connection_msg:
        env.server.client.set_connect(connection_msg)
        while True:
            data = await connection_msg.recv()
            json_data = json.loads(data)
            # print(json.dumps(json_data, indent=1))

async def main():
    env = Env()
    await asyncio.gather(connect_websocket(env))

if __name__ == "__main__":
    asyncio.run(main())