ACTION_NAMES = ["up", "down", "left", "right", "bomb", "detonate", "nop"]

def guess_action_based_on_gamestate_change(old_state, new_state, my_unit_id):
    # get old and new positions
    ox, oy = old_state['unit_state'][my_unit_id]['coordinates']
    nx, ny = new_state['unit_state'][my_unit_id]['coordinates']
    # check if we moved
    if ny > oy: return 0
    if ny < oy: return 1
    if nx > ox: return 3
    if nx < ox: return 2
    # where are our bombs on the board?
    old_bombs = set([str(e['x'])+','+str(e['y']) for e in old_state['entities'] if e.get('unit_id') == my_unit_id and e['type'] == 'b'])
    new_bombs = set([str(e['x'])+','+str(e['y']) for e in new_state['entities'] if e.get('unit_id') == my_unit_id and e['type'] == 'b'])
    # is there a new one?
    if len(new_bombs.difference(old_bombs)) > 0: return 4
    # where is fire caused by our bombs on the board?
    old_fire = set([str(e['x'])+','+str(e['y']) for e in old_state['entities'] if e.get('unit_id') == my_unit_id and e['type'] == 'x'])
    new_fire = set([str(e['x'])+','+str(e['y']) for e in new_state['entities'] if e.get('unit_id') == my_unit_id and e['type'] == 'x'])
    # is there new fire?
    if len(new_fire.difference(old_fire)) > 0: return 5
    # apparently, we did nothing
    return 6