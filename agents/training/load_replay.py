import json
import game_state
import copy

async def load_replay_file_as_trajectory(source_file):
    with open(source_file, 'rt') as f:
        # NOTE: we're replacing "owner_unit_id" with "unit_id" to make older replays from the 1068 server compatible, too.
        raw_data = f.read().replace("owner_unit_id","unit_id")
        data = json.loads(raw_data)
    # initialize a GameState object from the "initial_state" JSON packet
    game = game_state.GameState(None)
    game._state = data['payload']['initial_state']
    game._current_tick = game._state.get("tick")
    # initialize output array
    trajectory = []
    trajectory.append(copy.deepcopy(game._state))
    # get a list of all updates that the server sent. we will replay those
    update_packets = data['payload']['history']
    # do the packet replay
    while len(update_packets) > 0:
        # advance time
        game._current_tick += 1
        # apply all packets for that time
        while len(update_packets) > 0 and update_packets[0]['tick'] <= game._current_tick:
            await game._on_game_tick(update_packets[0])
            del update_packets[0]
        game._state['tick'] = game._current_tick
        # store the result
        trajectory.append(copy.deepcopy(game._state))
    return trajectory