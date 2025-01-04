"""This file is only for testing that we can use the sts_lightspeed library to run simulations on a game running with spirecomm"""
import json
from spirecomm.spire.game import Game
from spirecomm.communication.coordinator import Coordinator
from spirecomm.ai.agent import SimpleAgent
from spirecomm.spire.character import PlayerClass
from spirecomm.communication.action import *
from battle_scrum import BattleScumSearcher2
from mcts import MCTS
import slaythespire as sts
import logging
import os 
import time


dir_path = os.path.dirname(os.path.realpath(__file__))
logger = logging.getLogger(__name__)
logging.basicConfig(filename=os.path.join(dir_path, "logs", f"test-communicator_{time.time()}.log"), encoding='utf-8', level=logging.DEBUG)
logger.debug("Starting test-communicator")
class MyCoordinator(Coordinator):
    def __init__(self):
        super().__init__()
        self.logfile = None
        self.gc = None
        self.bc = None

    def receive_game_state_update(self, block=False, perform_callbacks=True):
        message = self.get_next_raw_message(block)

        if message is not None:
            communication_state = json.loads(message)
            logger.debug("Received message: %s", message)
            self.last_error = communication_state.get("error", None)
            self.game_is_ready = communication_state.get("ready_for_command")
            if self.last_error is None:
                self.in_game = communication_state.get("in_game")
                if self.in_game:
                    self.last_game_state = Game.from_json(communication_state.get("game_state"), communication_state.get("available_commands"))
                    try:
                        self.gc = sts.GameContext()
                        self.gc.init_from_json(message)
                        game_state = json.loads(message)
                        if "combat_state" in game_state["game_state"]:
                            self.bc = sts.BattleContext()
                            self.bc.init_from_json(self.gc, 
                                                json.dumps(game_state["game_state"]["combat_state"]))
                            self.last_game_state.bc = self.bc
                            logger.info("Battle Context loaded: %s", self.bc)
                    except Exception as e:
                        logger.exception(e, exc_info=True)
            if perform_callbacks:
                if self.last_error is not None:
                    self.action_queue.clear()
                    new_action = self.error_callback(self.last_error)
                    assert(isinstance(new_action, Action))
                    self.add_action_to_queue(new_action)
                elif self.in_game:
                    if len(self.action_queue) == 0 and perform_callbacks:
                        new_action = self.state_change_callback(self.last_game_state)
                        assert(isinstance(new_action, Action))
                        self.add_action_to_queue(new_action)
                elif self.stop_after_run:
                    self.clear_actions()
                else:
                    new_action = self.out_of_game_callback()
                    assert(isinstance(new_action, Action))
                    self.add_action_to_queue(new_action)
            return True
        return False

def get_comm_action_from_sts_action(sts_action, game_state):
    if sts_action.action_type == sts.SeachActionType.END_TURN:
        return EndTurnAction()
    if sts_action.action_type == sts.SeachActionType.CARD:
        target_idx = find_real_target_idx(sts_action, game_state)
        return PlayCardAction(card_index=sts_action.source_idx, target_index=target_idx)
    if sts_action.action_type == sts.SeachActionType.POTION:
        target_idx = find_real_target_idx(sts_action, game_state)
        return PotionAction(use=True, potion_index=sts_action.source_idx, target_index=target_idx)
    if sts_action.action_type == sts.SeachActionType.SINGLE_CARD_SELECT:
        return [ChooseAction(choice_index=sts_action.source_idx),
                OptionalCardSelectConfirmAction()]
    if sts_action.action_type == sts.SeachActionType.MULTI_CARD_SELECT:
        OptionalCardSelectConfirmAction()
    return None

def find_real_target_idx(sts_action, game_state):
    # TODO slime boss may be bugged here, will need to check
    target_idx = 0
    alive_target_idx = sts_action.target_idx
    for i in range(len(game_state.monsters)):
        if game_state.monsters[i].is_gone:
            target_idx += 1
        else:
            alive_target_idx -= 1
            if alive_target_idx < 0:
                break
    return target_idx


class TestAgent(SimpleAgent):
    def __init__(self):
        super().__init__(PlayerClass.IRONCLAD)
        self.state = None
        self.agent = sts.Agent()

    def handle_error(self, error):
        super().handle_error(error)

    def get_next_action_in_game(self, state):
        try:
            if hasattr(state, "bc") and state.bc is not None:
                #self.agent.playout_battle(bc)
                #bc.exit_battle(gc)
                searcher = sts.BattleAgent(state.bc)
                searcher.search(10000)
                actions = searcher.best_action_sequence
                action = get_comm_action_from_sts_action(actions[0], state)
                assert(isinstance(action, Action))
                return action
            else:
                return super().get_next_action_in_game(state)
        except Exception as e:
            logger.exception(e, exc_info=True)
            return super().get_next_action_in_game(state)

    def get_next_action_out_of_game(self):
        return super().get_next_action_out_of_game()


class TestMyAgent(SimpleAgent):
    def __init__(self):
        super().__init__(PlayerClass.IRONCLAD)
        self.state = None
        self.agent = sts.Agent()

    def handle_error(self, error):
        super().handle_error(error)

    def get_next_action_in_game(self, state):
        try:
            if hasattr(state, "bc") and state.bc is not None:
                #mcts = MCTS(num_simulations=10000, chance_sampling_breath=4)
                #mcts.set_root(state.bc)
                #mcts.search()
                #action = mcts.best_action()
                searcher = BattleScumSearcher2(state.bc)
                searcher.search(10000)
                action = searcher.best_action_sequence[0]

                action = get_comm_action_from_sts_action(action, state)
                assert(isinstance(action, Action))
                return action
            else:
                return super().get_next_action_in_game(state)
        except Exception as e:
            logger.exception(e, exc_info=True)
            return super().get_next_action_in_game(state)

    def get_next_action_out_of_game(self):
        return super().get_next_action_out_of_game()


try:
    agent = TestMyAgent()
    coordinator = MyCoordinator()
    coordinator.signal_ready()
    coordinator.register_command_error_callback(agent.handle_error)
    coordinator.register_state_change_callback(agent.get_next_action_in_game)
    coordinator.register_out_of_game_callback(agent.get_next_action_out_of_game)

    coordinator.play_one_game(PlayerClass.IRONCLAD)
except Exception as e:
    logger.exception(e, exc_info=True)
