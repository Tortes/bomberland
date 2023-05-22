def guess_reward_based_on_gamestate_change(old_state, new_state, my_unit_id):
    # get old and new state, and team ID
    old_unit_info = old_state['unit_state']
    new_unit_info = new_state['unit_state']
    my_team_id = new_unit_info[my_unit_id]['agent_id']
    # calculate how much HP each unit lost, while clipping HP to 0
    hp_diff = [(k, new_unit_info[k]['agent_id'], max(0, int(new_unit_info[k]['hp'])) - max(0, int(old_unit_info[k]['hp'])),) for k in new_unit_info]
    # sum up the HP losses for each team
    team_hp_lost = sum([max(0, -d) for k, a, d in hp_diff if a == my_team_id])
    enemy_hp_lost = sum([max(0, -d) for k, a, d in hp_diff if a != my_team_id])
    # the more damage the enemy takes, the better the reward
    reward = (enemy_hp_lost - team_hp_lost) / 3.0
    # check if we died, which would mean the unit trajectory ends here
    game_over = (new_unit_info[my_unit_id]['hp'] <= 0)
    return reward, game_over