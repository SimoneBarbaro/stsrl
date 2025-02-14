"""Script to test everything is working fine"""
import json
import logging
import os
import time

import numpy as np
import stsrl.slaythespire as sts

from stsrl.game_encoding import StsEncodings
from stsrl.mcts.battle_scrum import BattleScumSearcher2

logger = logging.getLogger(__name__)
logging.basicConfig(encoding='utf-8', level=logging.DEBUG)
logger.debug("Starting test-communicator")
dir_path = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(dir_path, "resources", "test-save.json")) as f:
    game_string = f.read()

game_state = json.loads(game_string)

print(game_state)

gc = sts.GameContext()

gc.init_from_json(game_string)
print(gc)
agent = sts.Agent()

agent.print_actions = True
agent.print_logs = True

game_state = game_state["game_state"] if "game_state" in game_state else game_state

if "combat_state" in game_state:
    bc = sts.BattleContext()
    bc.init_from_json(gc, json.dumps(game_state))
    print(bc)
    enc1 = StsEncodings.encode_battle(gc, bc)
    num_simulations = 10000
    agent = sts.BattleAgent(bc)
    cpp_start = time.time()
    agent.search(num_simulations)
    cpp_end = time.time()
    action = agent.best_action_sequence[0]
    print(f"cpp agent found: {action.print_desc(bc)}")
    action.execute(bc)
    print(bc)
    enc2 = StsEncodings.encode_battle(gc, bc)

    print(np.linalg.norm(enc2 - enc1))

    bc.exit_battle(gc)

    bc = sts.BattleContext()
    bc.init_from_json(gc, json.dumps(game_state))
    searcher_start = time.time()
    searcher = BattleScumSearcher2(bc)
    searcher.search(num_simulations)
    searcher_end = time.time()
    action = searcher.best_action_sequence[0]
    print(f"python agent found: {action.print_desc(bc)}")
    print(searcher.root)
    searcher.update_root(action)

    searcher.search(num_simulations)
    action = searcher.best_action_sequence[0]
    searcher.update_root(action)

    print(f"CPP time: {cpp_end - cpp_start}; Python time: {searcher_end - searcher_start}")

if False:
    gc.skip_battles = True
    agent.playout(gc)
