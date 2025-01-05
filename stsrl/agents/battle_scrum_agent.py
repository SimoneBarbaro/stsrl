from stsrl.agents.base_sts_agent import BaseAgent
from stsrl.controller.game_manager import GameManager
from stsrl.mcts.battle_scrum import BattleScumSearcher2
import stsrl.slaythespire as sts
import logging

logger = logging.getLogger(__name__)


class BattleScrumAgent(BaseAgent):
    def __init__(self, num_simulations=10000):
        super().__init__()
        self.num_simulations = num_simulations
        self.searcher: BattleScumSearcher2 = None

    def _process_combat_state(self):
        logger.info("Battle agent begin processing combat state")
        current_bc: sts.BattleContext = self.state.bc
        if self.searcher is None or current_bc.is_same_rng_counters(self.searcher.root_state):
            logger.info("Battle agent resetting search tree")
            self.searcher = BattleScumSearcher2(current_bc)
        self.searcher.search(self.num_simulations)

    def _pick_next_action(self):
        if self.state.in_combat:
            action = self.searcher.get_best_action()
            self.searcher.update_root(action)
            return GameManager.get_comm_action_from_sts_action(self.state, action)
        # No action outside combat for this agent
        return None
