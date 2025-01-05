"""This file is only for testing that we can use the sts_lightspeed library to run simulations on a game running with spirecomm"""
from spirecomm.ai.agent import SimpleAgent
from spirecomm.spire.character import PlayerClass
from spirecomm.communication.action import *
from stsrl.controller.game_manager import GameManager
import stsrl.slaythespire as sts
import logging

logger = logging.getLogger(__name__)


class BaseAgent(SimpleAgent):
    def __init__(self):
        super().__init__(PlayerClass.IRONCLAD)
        self.state: GameManager = None

    def handle_error(self, error):
        logger.exception(f"Error from communication mod: {error}", exc_info=True)
        # super().handle_error(error)

    def _process_combat_state(self):
        pass

    def _process_game_state(self):
        pass

    def _pick_next_action(self):
        pass

    def get_next_action_in_game(self, state: GameManager):
        try:
            self.state = state
            if state.in_combat:
                assert (state.bc is not None)
                self._process_combat_state()
            else:
                self._process_game_state()
            return self._pick_next_action()
        except Exception as e:
            logger.exception(e, exc_info=True)
            return super().get_next_action_in_game(state)

    def get_next_action_out_of_game(self):
        raise NotImplementedError("Base Agent does not take actions out of game")
