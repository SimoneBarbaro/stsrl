import json
import random
from typing import Optional

import gymnasium
import gymnasium as gym
import numpy as np

from stsrl.environments.constants import PLAYER_CLASS, ASCENSION
from stsrl.game_encoding import StsEncodings
import stsrl.slaythespire as sts
import logging

logger = logging.getLogger(__name__)


class StsBattleEnvironment(gymnasium.Env):
    def __init__(self):
        super().__init__()
        self.gc = None
        self.bc = None
        # self.observation_space = gym.spaces.Box(np.zeros_like(StsEncodings.nniInstance.getBattleObservationMaximums()),
        #                                        np.array(StsEncodings.nniInstance.getBattleObservationMaximums()))
        self.observation_space = gym.spaces.Dict(
            {
                "battle": gym.spaces.Box(np.zeros_like(StsEncodings.nniInstance.getBattleObservationMaximums()),
                                         np.ones_like(StsEncodings.nniInstance.getBattleObservationMaximums())),
                "legal_actions_mask": gym.spaces.Box(
                    np.zeros(StsEncodings.encodingInstance.battle_action_space_size),
                    np.ones(StsEncodings.encodingInstance.battle_action_space_size)),
            }
        )

        self.action_space = gym.spaces.Discrete(StsEncodings.encodingInstance.battle_action_space_size)
        self.turn = 0

    def _get_obs(self):
        return {
            "battle": StsEncodings.encode_battle(self.gc,
                                                 self.bc) / StsEncodings.nniInstance.getBattleObservationMaximums(),
            "legal_actions_mask": np.array(self.bc.get_valid_actions_mask(),
                                           dtype=bool) if self.bc.outcome == sts.BattleOutcome.UNDECIDED else np.zeros(StsEncodings.encodingInstance.battle_action_space_size, dtype=bool)
        }

    def _get_info(self):
        return {
            "environment_info": {
                "turn": self.bc.turn,
                "playerHp": self.bc.player.hp,
            },
            "legal_actions": [StsEncodings.encode_battle_action(a) for a in
                              self.bc.get_available_actions()] if self.bc.outcome == sts.BattleOutcome.UNDECIDED else []
        }

    def set_gc(self, gc: sts.GameContext):
        self.gc = gc

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if options is not None and "bc_json" in options:
            self.bc = sts.BattleContext()
            self.bc.init_from_json(self.gc, json.dumps(options["bc_json"]))
        elif options is None or "from_gc" not in options:
            if seed is None:
                seed = random.randint(0, 10000)
            self.gc = sts.GameContext(PLAYER_CLASS, seed, ASCENSION)
            # logger.debug("Game context Reset -> %s", str(self.gc))
            self.gc.skip_battles = True
            which_battle = random.randint(0, 30)
            i = 0
            while True:
                action = random.choice(self.gc.get_available_actions())
                # logger.debug("Game context available actions-> %s",
                #             ";".join([a.print_desc(self.gc) for a in self.gc.get_available_actions()]))
                # logger.debug("Game context execute action-> %s", action.print_desc(self.gc))
                self.gc.execute(action)
                # logger.debug("Game context -> %s", str(self.gc))
                if self.gc.outcome != sts.GameOutcome.UNDECIDED:
                    return self.reset()
                if self.gc.cur_room == sts.Room.MONSTER or self.gc.cur_room == sts.Room.ELITE or self.gc.cur_room == sts.Room.BOSS:
                    i += 1
                    if i > which_battle:
                        self.gc.skip_battles = False
                if self.gc.screen_state == sts.ScreenState.BATTLE:
                    self.gc.cur_hp -= min(random.randint(0, self.gc.cur_hp), random.randint(0, self.gc.cur_hp))
                    break

        self.bc = sts.BattleContext()
        self.bc.init(self.gc)
        logger.debug("Game context Reset -> %s", str(self.gc))
        logger.debug("Battle context Reset -> %s", str(self.bc))

        if seed is not None:
            self.bc.randomize_rng_counters(seed)

        observation = self._get_obs()
        info = self._get_info()
        self.turn = -1

        return observation, info

    def step(self, action):
        actions = self.bc.get_available_actions()
        action = StsEncodings.decode_battle_action(action)
        if self.turn != self.bc.turn:
            self.turn = self.bc.turn
            logger.debug("Battle context -> %s", str(self.bc))
        logger.debug("Battle context available actions-> %s", ";".join([a.print_desc(self.bc) for a in actions]))
        logger.debug("Battle context execute action-> %s", action.print_desc(self.bc))

        # TODO unnecessary
        if action.value in [a.value for a in actions]:
            self.bc.execute(action)
        else:
            raise NotImplementedError(
                f"Somehow we got a invalid action: {action.print_desc(self.bc)}, expected: {[a.print_desc(self.bc) for a in actions]}")

        reward = 0
        truncated = False
        terminated = False
        if self.bc.outcome == sts.BattleOutcome.PLAYER_VICTORY:
            reward = (1 + (self.bc.player.hp / self.bc.player.max_hp) + len(self.bc.potions) / 5) / 3
            terminated = True
        elif self.bc.outcome == sts.BattleOutcome.PLAYER_LOSS:
            reward = -1
            terminated = True
        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self) -> str:
        return str(self.bc)
