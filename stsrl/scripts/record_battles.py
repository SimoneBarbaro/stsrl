"""Run a GameCoordinator to record gameplay from slaythespire for imitation learning"""
import argparse
import logging
import os

from spirecomm.spire.character import PlayerClass

from stsrl.agents.battle_recorder_agent import ImitationLoggerAgent
from stsrl.controller.game_manager import GameCoordinator

dir_path = os.path.dirname(os.path.realpath(__file__))
logger = logging.getLogger(__name__)
logging.basicConfig(filename=os.path.join(dir_path, "../../logs", f"imitation_recorder.log"),
                    level=logging.DEBUG, filemode="w")

if __name__ == "__main__":

    try:
        parser = argparse.ArgumentParser(
            prog='ImitationRecorderSTS',
            description='Runs a Coordinator that receive game states from slaythespire game and records the battles to file',
            )
        parser.add_argument('--dir', help="dir where to save the games", type=str)
        parser.add_argument('--ascension', help="ascension level", type=int, default=0)
        args = parser.parse_args()
        os.makedirs(args.dir, exist_ok=True)

        coordinator = GameCoordinator()
        coordinator.signal_ready()

        filename = f"battle_logs_A{args.ascension}.log"
        agent = ImitationLoggerAgent(os.path.join(args.dir, filename))
        coordinator.register_command_error_callback(agent.handle_error)
        coordinator.register_state_change_callback(agent.get_next_action_in_game)
        while True:
            coordinator.play_one_game(PlayerClass.IRONCLAD, ascension_level=args.ascension)
    except Exception as e:
        logger.exception(e, exc_info=True)
