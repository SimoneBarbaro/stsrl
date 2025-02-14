import json
import logging
import random
from typing import Optional

import gymnasium
import gymnasium as gym
import numpy as np

from stsrl.environments.constants import PLAYER_CLASS, ASCENSION
from stsrl.game_encoding import StsEncodings
import stsrl.slaythespire as sts

logger = logging.getLogger(__name__)


class StsGameEnvironment(gymnasium.Env):

    def __init__(self):
        self.gc = sts.GameContext()

        self.observation_space = gym.spaces.Box(np.zeros_like(StsEncodings.nniInstance.getObservationMaximums()),
                                                np.array(StsEncodings.nniInstance.getObservationMaximums()))
        self.action_space = gym.spaces.Discrete(StsEncodings.encodingInstance.game_action_space_size)

    def _get_obs(self):
        return StsEncodings.encode_game(self.gc)

    def _get_info(self):
        return {
            "environment_info": {
                "floor_num": self.gc.floor_num,
                "playerHp": self.gc.cur_hp,
                "gold": self.gc.gold,
            },
            "legal_actions": [StsEncodings.encode_game_action(a) for a in self.gc.get_available_actions()],
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        self.act = 0
        if seed is None:
            seed = random.randint(0, 10000)
        if options is not None:
            self.gc = sts.GameContext()
            sts.init_from_json(json.dumps(options))
        else:
            self.gc = sts.GameContext(PLAYER_CLASS, seed, ASCENSION)
        logger.debug("Game context Reset -> %s", str(self.gc))
        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step_out_of_combat(self, bc):
        bc.exit_battle(self.gc)
        reward = 0
        truncated = False
        if self.gc.outcome != sts.GameOutcome.UNDECIDED:
            reward = self.gc.get_final_score()
        return self._get_obs(), reward, self.gc.outcome != sts.GameOutcome.UNDECIDED, truncated, self._get_info()

    def is_battle(self):
        return self.gc.screen_state == sts.ScreenState.BATTLE

    def step(self, action):
        reward = 0
        truncated = False
        if self.gc.screen_state == sts.ScreenState.BATTLE:
            raise NotImplementedError("Game environment cannot execute a battle")

        actions = self.gc.get_available_actions()
        action = StsEncodings.decode_game_action(action)
        if self.act != self.gc.act:
            logger.debug("Act map: %s", str(self.gc.map))
            self.act = self.gc.act
        logger.debug("Game context available actions-> %s", ";".join([a.print_desc(self.gc) for a in actions]))
        logger.debug("Game context execute action-> %s", action.print_desc(self.gc))
        # TODO unnecessary
        if action.value_model in [a.value_model for a in actions]:
            self.gc.execute(action)
            logger.debug("Game context -> %s", str(self.gc))
        else:
            raise NotImplementedError(
                f"Somehow we got a invalid action: {action.print_desc(self.gc)}, expected: {[a.print_desc(self.gc) for a in actions]}")

        if self.gc.outcome != sts.GameOutcome.UNDECIDED:
            reward = self.gc.get_final_score() / 1000
        return self._get_obs(), reward, self.gc.outcome != sts.GameOutcome.UNDECIDED, truncated, self._get_info()

    def render(self) -> str:
        return str(self.gc)