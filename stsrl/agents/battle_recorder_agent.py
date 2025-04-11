import logging

from spirecomm.ai.agent import SimpleAgent

from stsrl.agents.base_sts_agent import BaseAgent

logger = logging.getLogger(__name__)


class ImitationLoggerAgent(BaseAgent):
    """Simple agent to log a series of action taken by the human player as it receive the states from the coordinator"""
    def __init__(self, log_file):
        super().__init__()
        self.log_file = log_file

    def _process_combat_state(self):
        if self.old_state is not None and not self.old_state.in_combat:
            with open(self.log_file, "a") as f:
                f.write(self.game.json_state + "\n")

    def _process_game_state(self):
        pass

    def _pick_next_action(self):
        return super(SimpleAgent).get_next_action_in_game(self.game)
