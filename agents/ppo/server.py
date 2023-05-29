from game_state import GameState

AGENT_MAP = {"agentA" : "a", "agentB" : "b"}

class Server():
    def __init__(self, server_port: str, agent: str):
        self.client = GameState("ws://{}/?role=agent&agentId={}&name=defaultName".format(server_port, agent))
        self.admin = GameState("ws://{}/?role=admin".format(server_port))
        self.state = None
        self.agnent = AGENT_MAP[agent]
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
        print("Get tick " + str(tick_number))
        self.agent = game_state["connection"]["agent_id"]
        self.unit = game_state["agents"][self.agent]["unit_ids"][0]
        self.tick = tick_number
        self.state = game_state
        print("Tick callback")