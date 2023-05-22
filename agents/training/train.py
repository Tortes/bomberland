import asyncio
import json
import os
import pty
import subprocess
import time
import docker
import websockets
from tqdm import tqdm

from prepare_server import *
from load_replay import *

SERVER_IP = '127.0.0.1'
SERVER_PORT = 3000
NUMBER_OF_REPLAYS = 16
CMD_AGENT_A = f'docker run -t --rm -e "GAME_CONNECTION_STRING=ws://{SERVER_IP}:{SERVER_PORT}/?role=agent&agentId=agentA&name=defaultName" public.docker.cloudgamepad.com/gocoder/oiemxoijsircj-round3sub-s1555'
CMD_AGENT_B = f'docker run -t --rm -e "GAME_CONNECTION_STRING=ws://{SERVER_IP}:{SERVER_PORT}/?role=agent&agentId=agentB&name=defaultName" public.docker.cloudgamepad.com/gocoder/oiemxoijsircj-round3sub-s1555'
VERBOSE = False

# Helper
docker_client = docker.from_env()
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
os.makedirs("./replays", exist_ok=True)

def readline_skip_empty(reader):
    while True:
        l = reader.readline().strip()
        if len(l) > 0:
            return l
        

# Sample the Replays 
async def sample_replay():
    for idx in tqdm(range(NUMBER_OF_REPLAYS)):
        # spawn server with docker and get admin connection
        container = docker_client.containers.run('coderone.azurecr.io/bomberland-engine:2381', detach=True, remove=True, environment=['TRAINING_MODE_ENABLED=1'], ports={'3000/tcp': SERVER_PORT})
        admin_connection = await get_write_only_admin_connection(SERVER_PORT, VERBOSE)
        # spawn agent A
        a1read, a1write = pty.openpty()
        subprocess.Popen(CMD_AGENT_A, shell=True, stdout=a1write, close_fds=True)
        a1out = os.fdopen(a1read)
        # spawn agent B
        a2read, a2write = pty.openpty()
        subprocess.Popen(CMD_AGENT_B, shell=True, stdout=a2write, close_fds=True)
        a2out = os.fdopen(a2read)
        # run game loop
        async def run_game_loop():
            while True:
                # read output from agent A
                while True:
                    l = readline_skip_empty(a1out)
                    if VERBOSE: print("A1>",l)
                    if l.startswith('===TICK FINISHED'): break
                    if l == '===ENDGAME_STATE===': return json.loads(readline_skip_empty(a1out))
                # read output from agent B
                while True:
                    l = readline_skip_empty(a2out)
                    if VERBOSE: print("A2>",l)
                    if l.startswith('===TICK FINISHED'): break
                    if l == '===ENDGAME_STATE===': return json.loads(readline_skip_empty(a2out))
                # short wait, then ask server to tick using admin connection
                time.sleep(0.1)
                await admin_connection.send('{"type": "request_tick"}')
        end_state = await run_game_loop()
        # write replay file
        with open(f'./replays/{idx}.json', 'wt') as f:
            f.write(json.dumps(end_state, indent=4))
        # shutdown server
        container.stop()



if (__name__ == "__main__"):
    print(r"<<<<<<<<<<<<<<<<<<<")
    # asyncio.run(sample_replay())
    asyncio.run(load_replay_file_as_trajectory())
