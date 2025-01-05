"""Classes for managing spirecomm game coordination wit sts_lightspeed object initialization"""
import json
from spirecomm.spire.game import Game
from spirecomm.communication.coordinator import Coordinator
from spirecomm.communication.action import *
import stsrl.slaythespire as sts
import logging


logger = logging.getLogger(__name__)


class GameManager(Game):
    """A class to manage the spirecomm game as well as the simulation"""
    def __init__(self):
        super().__init__()
        self.gc: sts.GameContext = None
        self.bc: sts.BattleContext = None

    @classmethod
    def from_json(cls, json_state, available_commands):
        logger.debug(f"Received game state to initialize {json_state}")
        game = super().from_json(json_state, available_commands)
        game.gc = sts.GameContext()
        game_str = json.dumps(json_state)
        logger.debug(f"Creating game context from: {game_str}")
        game.gc.init_from_json(game_str)
        logger.debug(f"Initialized game state from json: {game.gc}")

        if "combat_state" in json_state:
            game.bc = sts.BattleContext()
            game.bc.init_from_json(game.gc, game_str)
            logger.debug(f"Initialized combat state from json: {game.bc}")
        return game

    @staticmethod
    def find_real_target_idx(self, sts_action: sts.SearchAction):
        """sts_lightspeed removes dead monsters so target index is not alligned, here we recover true target index"""
        # TODO slime boss may be bugged here, will need to check
        target_idx = 0
        alive_target_idx = sts_action.target_idx
        for i in range(len(self.monsters)):
            if self.monsters[i].is_gone:
                target_idx += 1
            else:
                alive_target_idx -= 1
                if alive_target_idx < 0:
                    break
        return target_idx

    @staticmethod
    def get_comm_action_from_sts_action(self, sts_action: sts.SearchAction) -> Action:
        """Convert an action from sts_ligthspeed SearchAction to spirecomm Action"""
        if sts_action.action_type == sts.SeachActionType.END_TURN:
            return EndTurnAction()
        if sts_action.action_type == sts.SeachActionType.CARD:
            target_idx = GameManager.find_real_target_idx(self, sts_action)
            return PlayCardAction(card_index=sts_action.source_idx, target_index=target_idx)
        if sts_action.action_type == sts.SeachActionType.POTION:
            target_idx = GameManager.find_real_target_idx(self, sts_action)
            return PotionAction(use=True, potion_index=sts_action.source_idx, target_index=target_idx)
        if sts_action.action_type == sts.SeachActionType.SINGLE_CARD_SELECT:
            return ChooseAction(choice_index=sts_action.source_idx)
        if sts_action.action_type == sts.SeachActionType.MULTI_CARD_SELECT:
            OptionalCardSelectConfirmAction()
        raise NotImplementedError(f"We got unexpected action: {sts_action.print_desc()} for combat state {self.bc}")


class GameCoordinator(Coordinator):
    """I need this coordinator to create a GameManager instead of a game object
    The code otherwise is the same but we need to copy it because it's not possible to just substitute the class we create.
    Also, this coordinator does not take orders for out of game actions"""

    def receive_game_state_update(self, block=False, perform_callbacks=True):
        """Using the next message from Communication Mod, update the stored game state

        :param block: set to True to wait for the next message
        :type block: bool
        :param perform_callbacks: set to True to perform callbacks based on the new game state
        :type perform_callbacks: bool
        :return: whether a message was received
        """
        message = self.get_next_raw_message(block)
        if message is not None:
            communication_state = json.loads(message)
            self.last_error = communication_state.get("error", None)
            self.game_is_ready = communication_state.get("ready_for_command")
            if self.last_error is None:
                self.in_game = communication_state.get("in_game")
                if self.in_game:
                    self.last_game_state = GameManager.from_json(communication_state.get("game_state"), communication_state.get("available_commands"))
            if perform_callbacks:
                if self.last_error is not None:
                    self.action_queue.clear()
                    if self.error_callback is not None:
                        new_action = self.error_callback(self.last_error)
                        if isinstance(new_action, Action):
                            self.add_action_to_queue(new_action)
                elif self.in_game:
                    if len(self.action_queue) == 0 and perform_callbacks:
                        if self.state_change_callback is not None:
                            new_action = self.state_change_callback(self.last_game_state)
                            if new_action is None:
                                logger.warning("Agent failed to propose action")
                            else:
                                logger.debug(f"Agent finished thinking, proposing action: {new_action}")
                            if isinstance(new_action, Action):
                                self.add_action_to_queue(new_action)
                elif self.stop_after_run:
                    self.clear_actions()
                # We don't perform out of game callbacks
                #else:
                #    new_action = self.out_of_game_callback()
                #    self.add_action_to_queue(new_action)
            return True
        return False
