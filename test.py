import time
from mcts import MCTS
from battle_scrum import BattleScumSearcher2
import slaythespire as sts

game_string = """
{"available_commands":["play","end","key","click","wait","state"],"ready_for_command":true,"in_game":true,"game_state":{"screen_type":"NONE","screen_state":{},"seed":1639520189346654817,"combat_state":{"draw_pile":[{"exhausts":false,"is_playable":true,"cost":1,"name":"Defend","id":"Defend_R","type":"SKILL","ethereal":false,"uuid":"5fe3d10e-06b3-4dd0-bf51-807b6175f8bb","upgrades":0,"rarity":"BASIC","has_target":false},{"exhausts":false,"is_playable":true,"cost":1,"name":"Strike","id":"Strike_R","type":"ATTACK","ethereal":false,"uuid":"d4dde391-0799-4588-9386-b4e8a8d975c9","upgrades":0,"rarity":"BASIC","has_target":true},{"exhausts":false,"is_playable":true,"cost":1,"name":"Defend","id":"Defend_R","type":"SKILL","ethereal":false,"uuid":"05878ab5-4ca3-45a6-a22b-9e1a8fbb572c","upgrades":0,"rarity":"BASIC","has_target":false},{"exhausts":false,"is_playable":true,"cost":1,"name":"Strike","id":"Strike_R","type":"ATTACK","ethereal":false,"uuid":"65463559-31cc-4a1e-9e50-bf4cbc6ce9a7","upgrades":0,"rarity":"BASIC","has_target":true},{"exhausts":false,"is_playable":true,"cost":1,"name":"Strike","id":"Strike_R","type":"ATTACK","ethereal":false,"uuid":"aceba5b7-d99e-4400-a2ed-e20267a7686b","upgrades":0,"rarity":"BASIC","has_target":true}],"discard_pile":[],"exhaust_pile":[],"cards_discarded_this_turn":0,"times_damaged":0,"monsters":[{"is_gone":false,"move_hits":1,"move_base_damage":11,"half_dead":false,"move_adjusted_damage":-1,"max_hp":42,"intent":"DEBUG","move_id":1,"name":"Jaw Worm","current_hp":42,"block":0,"id":"JawWorm","powers":[]}],"turn":1,"limbo":[],"hand":[{"exhausts":false,"is_playable":true,"cost":1,"name":"Defend","id":"Defend_R","type":"SKILL","ethereal":false,"uuid":"fa84e664-3e10-43a3-9e9c-a608f5e61881","upgrades":0,"rarity":"BASIC","has_target":false},{"exhausts":false,"is_playable":true,"cost":1,"name":"Strike","id":"Strike_R","type":"ATTACK","ethereal":false,"uuid":"c0b1badf-76a0-4cfb-b5f3-6c159877f26a","upgrades":0,"rarity":"BASIC","has_target":true},{"exhausts":false,"is_playable":true,"cost":1,"name":"Strike","id":"Strike_R","type":"ATTACK","ethereal":false,"uuid":"0e53f160-4815-4079-9aeb-bb602de0912c","upgrades":0,"rarity":"BASIC","has_target":true},{"exhausts":false,"is_playable":true,"cost":1,"name":"Defend","id":"Defend_R","type":"SKILL","ethereal":false,"uuid":"3ecafdb6-caac-46f3-8e55-0753ff06189a","upgrades":0,"rarity":"BASIC","has_target":false},{"exhausts":false,"is_playable":true,"cost":2,"name":"Bash","id":"Bash","type":"ATTACK","ethereal":false,"uuid":"bfaf51cc-932c-482d-aa69-0bbf11798e9e","upgrades":0,"rarity":"BASIC","has_target":true}],"player":{"orbs":[],"current_hp":80,"block":0,"max_hp":80,"powers":[],"energy":3}},"deck":[{"exhausts":false,"is_playable":true,"cost":1,"name":"Strike","id":"Strike_R","type":"ATTACK","ethereal":false,"uuid":"c0b1badf-76a0-4cfb-b5f3-6c159877f26a","upgrades":0,"rarity":"BASIC","has_target":true},{"exhausts":false,"is_playable":true,"cost":1,"name":"Strike","id":"Strike_R","type":"ATTACK","ethereal":false,"uuid":"65463559-31cc-4a1e-9e50-bf4cbc6ce9a7","upgrades":0,"rarity":"BASIC","has_target":true},{"exhausts":false,"is_playable":true,"cost":1,"name":"Strike","id":"Strike_R","type":"ATTACK","ethereal":false,"uuid":"d4dde391-0799-4588-9386-b4e8a8d975c9","upgrades":0,"rarity":"BASIC","has_target":true},{"exhausts":false,"is_playable":true,"cost":1,"name":"Strike","id":"Strike_R","type":"ATTACK","ethereal":false,"uuid":"0e53f160-4815-4079-9aeb-bb602de0912c","upgrades":0,"rarity":"BASIC","has_target":true},{"exhausts":false,"is_playable":true,"cost":1,"name":"Strike","id":"Strike_R","type":"ATTACK","ethereal":false,"uuid":"aceba5b7-d99e-4400-a2ed-e20267a7686b","upgrades":0,"rarity":"BASIC","has_target":true},{"exhausts":false,"is_playable":true,"cost":1,"name":"Defend","id":"Defend_R","type":"SKILL","ethereal":false,"uuid":"fa84e664-3e10-43a3-9e9c-a608f5e61881","upgrades":0,"rarity":"BASIC","has_target":false},{"exhausts":false,"is_playable":true,"cost":1,"name":"Defend","id":"Defend_R","type":"SKILL","ethereal":false,"uuid":"3ecafdb6-caac-46f3-8e55-0753ff06189a","upgrades":0,"rarity":"BASIC","has_target":false},{"exhausts":false,"is_playable":true,"cost":1,"name":"Defend","id":"Defend_R","type":"SKILL","ethereal":false,"uuid":"05878ab5-4ca3-45a6-a22b-9e1a8fbb572c","upgrades":0,"rarity":"BASIC","has_target":false},{"exhausts":false,"is_playable":true,"cost":1,"name":"Defend","id":"Defend_R","type":"SKILL","ethereal":false,"uuid":"5fe3d10e-06b3-4dd0-bf51-807b6175f8bb","upgrades":0,"rarity":"BASIC","has_target":false},{"exhausts":false,"is_playable":true,"cost":2,"name":"Bash","id":"Bash","type":"ATTACK","ethereal":false,"uuid":"bfaf51cc-932c-482d-aa69-0bbf11798e9e","upgrades":0,"rarity":"BASIC","has_target":true}],"relics":[{"name":"Burning Blood","id":"Burning Blood","counter":-1}],"max_hp":80,"act_boss":"Slime Boss","gold":99,"action_phase":"WAITING_ON_USER","act":1,"screen_name":"NONE","room_phase":"COMBAT","is_screen_up":false,"potions":[{"requires_target":false,"can_use":false,"can_discard":false,"name":"Potion Slot","id":"Potion Slot"},{"requires_target":false,"can_use":false,"can_discard":false,"name":"Potion Slot","id":"Potion Slot"},{"requires_target":false,"can_use":false,"can_discard":false,"name":"Potion Slot","id":"Potion Slot"}],"current_hp":80,"floor":1,"ascension_level":0,"class":"IRONCLAD","map":[{"symbol":"M","children":[{"x":0,"y":1},{"x":2,"y":1}],"x":1,"y":0,"parents":[]},{"symbol":"M","children":[{"x":4,"y":1}],"x":3,"y":0,"parents":[]},{"symbol":"M","children":[{"x":5,"y":1}],"x":4,"y":0,"parents":[]},{"symbol":"$","children":[{"x":0,"y":2}],"x":0,"y":1,"parents":[]},{"symbol":"?","children":[{"x":1,"y":2},{"x":3,"y":2}],"x":2,"y":1,"parents":[]},{"symbol":"?","children":[{"x":5,"y":2}],"x":4,"y":1,"parents":[]},{"symbol":"M","children":[{"x":5,"y":2}],"x":5,"y":1,"parents":[]},{"symbol":"M","children":[{"x":0,"y":3}],"x":0,"y":2,"parents":[]},{"symbol":"$","children":[{"x":0,"y":3}],"x":1,"y":2,"parents":[]},{"symbol":"M","children":[{"x":3,"y":3}],"x":3,"y":2,"parents":[]},{"symbol":"?","children":[{"x":4,"y":3},{"x":6,"y":3}],"x":5,"y":2,"parents":[]},{"symbol":"M","children":[{"x":0,"y":4},{"x":1,"y":4}],"x":0,"y":3,"parents":[]},{"symbol":"?","children":[{"x":3,"y":4}],"x":3,"y":3,"parents":[]},{"symbol":"?","children":[{"x":4,"y":4}],"x":4,"y":3,"parents":[]},{"symbol":"M","children":[{"x":6,"y":4}],"x":6,"y":3,"parents":[]},{"symbol":"M","children":[{"x":0,"y":5}],"x":0,"y":4,"parents":[]},{"symbol":"$","children":[{"x":1,"y":5}],"x":1,"y":4,"parents":[]},{"symbol":"M","children":[{"x":3,"y":5}],"x":3,"y":4,"parents":[]},{"symbol":"M","children":[{"x":5,"y":5}],"x":4,"y":4,"parents":[]},{"symbol":"M","children":[{"x":5,"y":5}],"x":6,"y":4,"parents":[]},{"symbol":"E","children":[{"x":0,"y":6}],"x":0,"y":5,"parents":[]},{"symbol":"R","children":[{"x":0,"y":6}],"x":1,"y":5,"parents":[]},{"symbol":"M","children":[{"x":4,"y":6}],"x":3,"y":5,"parents":[]},{"symbol":"M","children":[{"x":5,"y":6}],"x":5,"y":5,"parents":[]},{"symbol":"M","children":[{"x":0,"y":7},{"x":1,"y":7}],"x":0,"y":6,"parents":[]},{"symbol":"E","children":[{"x":3,"y":7}],"x":4,"y":6,"parents":[]},{"symbol":"M","children":[{"x":5,"y":7}],"x":5,"y":6,"parents":[]},{"symbol":"E","children":[{"x":0,"y":8}],"x":0,"y":7,"parents":[]},{"symbol":"?","children":[{"x":1,"y":8}],"x":1,"y":7,"parents":[]},{"symbol":"M","children":[{"x":2,"y":8}],"x":3,"y":7,"parents":[]},{"symbol":"R","children":[{"x":4,"y":8},{"x":5,"y":8}],"x":5,"y":7,"parents":[]},{"symbol":"T","children":[{"x":0,"y":9}],"x":0,"y":8,"parents":[]},{"symbol":"T","children":[{"x":1,"y":9},{"x":2,"y":9}],"x":1,"y":8,"parents":[]},{"symbol":"T","children":[{"x":3,"y":9}],"x":2,"y":8,"parents":[]},{"symbol":"T","children":[{"x":4,"y":9}],"x":4,"y":8,"parents":[]},{"symbol":"T","children":[{"x":4,"y":9}],"x":5,"y":8,"parents":[]},{"symbol":"?","children":[{"x":0,"y":10}],"x":0,"y":9,"parents":[]},{"symbol":"M","children":[{"x":2,"y":10}],"x":1,"y":9,"parents":[]},{"symbol":"E","children":[{"x":2,"y":10}],"x":2,"y":9,"parents":[]},{"symbol":"?","children":[{"x":4,"y":10}],"x":3,"y":9,"parents":[]},{"symbol":"R","children":[{"x":4,"y":10},{"x":5,"y":10}],"x":4,"y":9,"parents":[]},{"symbol":"R","children":[{"x":1,"y":11}],"x":0,"y":10,"parents":[]},{"symbol":"?","children":[{"x":1,"y":11}],"x":2,"y":10,"parents":[]},{"symbol":"?","children":[{"x":3,"y":11},{"x":4,"y":11}],"x":4,"y":10,"parents":[]},{"symbol":"M","children":[{"x":6,"y":11}],"x":5,"y":10,"parents":[]},{"symbol":"?","children":[{"x":0,"y":12},{"x":2,"y":12}],"x":1,"y":11,"parents":[]},{"symbol":"R","children":[{"x":2,"y":12}],"x":3,"y":11,"parents":[]},{"symbol":"?","children":[{"x":3,"y":12}],"x":4,"y":11,"parents":[]},{"symbol":"R","children":[{"x":6,"y":12}],"x":6,"y":11,"parents":[]},{"symbol":"R","children":[{"x":0,"y":13},{"x":1,"y":13}],"x":0,"y":12,"parents":[]},{"symbol":"M","children":[{"x":1,"y":13}],"x":2,"y":12,"parents":[]},{"symbol":"M","children":[{"x":4,"y":13}],"x":3,"y":12,"parents":[]},{"symbol":"M","children":[{"x":5,"y":13}],"x":6,"y":12,"parents":[]},{"symbol":"?","children":[{"x":0,"y":14}],"x":0,"y":13,"parents":[]},{"symbol":"M","children":[{"x":0,"y":14},{"x":2,"y":14}],"x":1,"y":13,"parents":[]},{"symbol":"E","children":[{"x":3,"y":14}],"x":4,"y":13,"parents":[]},{"symbol":"M","children":[{"x":5,"y":14}],"x":5,"y":13,"parents":[]},{"symbol":"R","children":[{"x":3,"y":16}],"x":0,"y":14,"parents":[]},{"symbol":"R","children":[{"x":3,"y":16}],"x":2,"y":14,"parents":[]},{"symbol":"R","children":[{"x":3,"y":16}],"x":3,"y":14,"parents":[]},{"symbol":"R","children":[{"x":3,"y":16}],"x":5,"y":14,"parents":[]}],"room_type":"MonsterRoom"}}
"""

import json
game_state = json.loads(game_string)

print(game_state)

gc = sts.GameContext()

gc.init_from_json(game_string)
print(gc)
agent = sts.Agent()

agent.print_actions = True
agent.print_logs = True

if "combat_state" in game_state["game_state"]:
    bc = sts.BattleContext()
    bc.init_from_json(gc, json.dumps(game_state["game_state"]["combat_state"]))
    print(bc)
    num_simulations = 10000
    #agent.playout_battle(bc)
    mcts_start = time.time()
    mcts = MCTS(num_simulations=num_simulations)
    mcts.set_root(bc)
    mcts.search()
    mcts_end = time.time()
    action = mcts.best_action()
    action.execute(bc)
    print(bc)
    
    bc.exit_battle(gc)

    bc = sts.BattleContext()
    bc.init_from_json(gc, json.dumps(game_state["game_state"]["combat_state"]))
    searcher_start = time.time()
    searcher = BattleScumSearcher2(bc)
    searcher.search(num_simulations)
    searcher_end = time.time()

    print(f"MCTS time: {mcts_end - mcts_start}; Searcher time: {searcher_end - searcher_start}")

# gc.skip_battles = True

#agent.playout(gc)