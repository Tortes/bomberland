import numpy as np
from env import Env
INVULERABILITY_DURATION = 5.0

class State():
    def __init__(self, env: Env):
        self.env = env
        self.unit_layers = np.zeros([15,15,4,2], np.float32)
        self.entity_hp_layers = np.zeros([15,15,5], np.uint8)
        self.entity_pos_layers = np.zeros([15,15], np.float32)
        self.entity_duration_layers = np.zeros([15,15], np.float32)
        self.bomb_expire_layers = np.zeros([15,15], np.float32)
        self.world_layers = np.ones([15,15], np.float32)
        self.tick = 0

    def update_state(self, state_info, tick_number):
        self.tick = tick_number

    def update_unit_layers(self, state_info):
        agent2units_single = {
            'x' : self.env.server.unit,
            'a' : [u for u in ['c'] if u != self.env.server.unit],
            'b': [u for u in ['d'] if u != self.env.server.unit],
        }
        agent_order = ['a','b']
        agent_order.sort(key=lambda x: int(x != self.env.server.agnet))
        agent_order_full = ['x'] + agent_order

        for agent in agent_order_full:
            tmp = np.zeros([15,15,4], np.float32)
            for unit_id in agent2units_single[agent]:
                unit = agent2units_single[unit_id]
                cux, cuy = unit['coordinates']
                tmp[cuy, cux, 0] = 1.0
                tmp[cuy, cux, 1] = float(max(0,unit['hp']))
                tmp[cuy, cux, 2] = float(max(0,unit['invulnerability'] - self.tick)) / INVULERABILITY_DURATION
                tmp[cuy, cux, 3] = min(float(unit['blast_diameter']) / 3.0, 7)
            idx = agent_order.index(self.env.server.agent)
            self.unit_layers[:,:,0,idx] = (f'agent {agent} positions', tmp[:,:,0], '01.0f')
            self.unit_layers[:,:,1,idx] = (f'agent {agent} HP', tmp[:,:,0], '01.0f')
            self.unit_layers[:,:,2,idx] = (f'agent {agent} invulnerability', tmp[:,:,0], '3.1f')
            self.unit_layers[:,:,3,idx] = (f'agent {agent} blast_diameter', tmp[:,:,0], '01.0f')
        print(self.unit_layers)

    def update_entity_layers(self, state_info):
        tiles = np.zeros([15, 15], np.uint8)
        for e in state_info["entities"]:
            type = e['type']
            x, y = e['x'], e['y']
            if type in ['m', 'o', 'w']:
                tiles[y, x] = e.get('hp', 99)
            elif type in ['x']:
                tiles[y, x] = 100 + max(0, e.get('expires', self.tick + 99) - self.tick - 1)
        self.entity_hp_layers[:,:,0] = (f'environment HP 1', np.float32(tiles == 1), '01.0f')
        self.entity_hp_layers[:,:,1] = (f'environment HP 2', np.float32(tiles == 2), '01.0f')
        self.entity_hp_layers[:,:,2] = (f'environment HP 3', np.float32(tiles == 3), '01.0f')
        self.entity_hp_layers[:,:,3] = (f'environment HP 99', np.float32(tiles == 99), '01.0f')

        fire_time = np.maximum(np.float32(tiles) - 100, np.zeros_like(tiles)) / 100.0
        self.entity_hp_layers[:,:,4] = (f'fire time', np.float32(fire_time), '3.1f')
        print(self.entity_hp_layers)

        for type in ['b', 'a', 'bp', 'fp']:
            layer = np.zeros([15, 15], np.float32)
            for e in entities:
                if e['type'] != type: continue
                layer[e['y'], e['x']] = 1.0
            layers.append((f'entity {type} pos', layer, '01.0f'))

    def update_entity_pos(self, state_info):

