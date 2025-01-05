"""This file is only for testing that we can use the sts_lightspeed library to run simulations on a game running with spirecomm"""
import logging
import os

from spirecomm.spire.character import PlayerClass

from stsrl.agents.battle_scrum_agent import BattleScrumAgent
from stsrl.agents.sts_battle_scrum_agent import BattleAgent
from stsrl.controller.game_manager import GameCoordinator

dir_path = os.path.dirname(os.path.realpath(__file__))
logger = logging.getLogger(__name__)
logging.basicConfig(filename=os.path.join(dir_path, "logs", f"test-communicator.log"),
                    level=logging.DEBUG)

if __name__ == "__main__":

    try:

        agent = BattleAgent()
        coordinator = GameCoordinator()
        coordinator.signal_ready()
        coordinator.register_command_error_callback(agent.handle_error)
        coordinator.register_state_change_callback(agent.get_next_action_in_game)
        #coordinator.register_out_of_game_callback(agent.get_next_action_out_of_game)

        coordinator.play_one_game(PlayerClass.IRONCLAD, seed="3A8HJ4HRCV2A5")
    except Exception as e:
        logger.exception(e, exc_info=True)
