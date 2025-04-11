import random
import json
import logging
from typing import Optional

import gymnasium
import gymnasium as gym
import numpy as np

from stsrl.environments.constants import PLAYER_CLASS, ASCENSION
from stsrl.game_encoding import StsEncodings
import stsrl.slaythespire as sts

logger = logging.getLogger(__name__)


class StsEnvironment(gymnasium.Env):
    def __init__(self, config_json=None, ascension_level=0):
        self.gc = None
        self.bc = None
        self.config_json = config_json
        self.turn = -1
        self.act = 0
        self.ascension_level = ascension_level

        self.observation_space = gym.spaces.Dict(
            {
                # There is probably a better way to describe the environment
                "game": gym.spaces.Box(np.zeros_like(StsEncodings.nniInstance.getObservationMaximums()).astype("float32"),
                                       np.ones_like(StsEncodings.nniInstance.getObservationMaximums()).astype("float32")),
                "battle": gym.spaces.Box(np.zeros_like(StsEncodings.nniInstance.getBattleObservationMaximums()).astype("float32"),
                                         np.ones_like(StsEncodings.nniInstance.getBattleObservationMaximums()).astype("float32")),
                "legal_actions_mask": gym.spaces.MultiBinary(StsEncodings.encodingInstance.battle_action_space_size),
                "is_battle": gym.spaces.MultiBinary(1)
            }
        )
        self.action_space = gym.spaces.Discrete(StsEncodings.encodingInstance.battle_action_space_size)
        self.give_continuous_reward = False

    def _get_valid_action_mask(self):
        if self.gc.outcome != sts.GameOutcome.UNDECIDED:
            return np.zeros(StsEncodings.encodingInstance.battle_action_space_size, dtype=bool)
        if self.gc.screen_state == sts.ScreenState.BATTLE:
            return np.array(self.bc.get_valid_actions_mask(), dtype=bool)
        return np.concatenate([np.array(self.gc.get_valid_actions_mask(), dtype=bool),
                               np.zeros(
                                   StsEncodings.encodingInstance.battle_action_space_size - StsEncodings.encodingInstance.game_action_space_size)])

    def _get_obs(self):
        if self.gc.screen_state == sts.ScreenState.BATTLE:
            return {
                "game": StsEncodings.encode_game(self.gc) / StsEncodings.nniInstance.getObservationMaximums(),
                "battle": StsEncodings.encode_battle(self.gc,
                                                     self.bc) / StsEncodings.nniInstance.getBattleObservationMaximums(),
                "legal_actions_mask": self._get_valid_action_mask(),
                "is_battle": np.array([True])
            }
        else:
            return {
                "game": StsEncodings.encode_game(self.gc) / StsEncodings.nniInstance.getObservationMaximums(),
                "battle": np.zeros(StsEncodings.nniInstance.battle_space_size, dtype=np.float32),
                "legal_actions_mask": self._get_valid_action_mask(),
                "is_battle": np.array([False])
            }

    def _get_info(self):
        return {
            "legal_actions": [StsEncodings.encode_battle_action(a) for a in
                              self.bc.get_available_actions()] if self.is_battle() else [
                StsEncodings.encode_game_action(a) for a in self.gc.get_available_actions()]
        }

    def set_state(self, gc=None, bc=None):
        self.gc = gc
        self.bc = bc
        self.turn = -1
        self.act = 0

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super(StsEnvironment, self).reset(seed=seed)
        if seed is None:
            seed = random.randint(0, 10000)
        if self.config_json is not None:
            self.gc = sts.GameContext()
            self.gc.init_from_json(self.config_json)

            # Reset starts from battle?
            if self.is_battle():
                self.bc = sts.BattleContext()
                self.turn = -1
                game_state = json.loads(self.config_json)
                game_state = game_state["game_state"] if "game_state" in game_state else game_state
                if "combat_state" in game_state:
                    self.bc.init_from_json(self.gc, json.dumps(game_state))
                else:
                    self.bc.init(self.gc)

        elif options is not None and len(options) > 0:
            self.gc = sts.GameContext()
            self.gc.init_from_json(json.dumps(options))
        else:
            self.gc = sts.GameContext(PLAYER_CLASS, seed, self.ascension_level)

        logger.debug("Game context Reset -> %s", str(self.gc))
        observation = self._get_obs()
        info = self._get_info()
        self.turn = -1
        self.act = 0

        return observation, info

    def _battle_step(self, action):
        action = StsEncodings.decode_battle_action(action)
        if action.value in [a.value for a in self.bc.get_available_actions()]:
            if self.turn != self.bc.turn:
                self.turn = self.bc.turn
                logger.debug("Battle context -> %s", str(self.bc))
            logger.debug("Battle context available actions-> %s",
                         ";".join([a.print_desc(self.bc) for a in self.bc.get_available_actions()]))
            logger.debug("Battle context execute action-> %s", action.print_desc(self.bc))

            self.bc.execute(action)

    def is_battle(self):
        return self.gc.screen_state == sts.ScreenState.BATTLE

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = np.argmax(action)
        reward = 0
        truncated = False
        if self.is_battle():
            try:
                self._battle_step(action)
            except IndexError as e:
                logger.error("Invalid action %s submitted: %s", str(action), e)
                return self._get_obs(), reward, self.gc.outcome != sts.GameOutcome.UNDECIDED, truncated, self._get_info()

            if self.bc.is_terminal():
                logger.debug("Battle context end state -> %s", str(self.bc))
                if self.bc.outcome == sts.BattleOutcome.PLAYER_LOSS:
                    reward = -1
                else:
                    reward = (self.bc.player.hp / self.bc.player.max_hp) + (self.bc.player.max_hp / self.gc.max_hp - 1)
                self.bc.exit_battle(self.gc)

        else:
            try:
                action = StsEncodings.decode_game_action(action)
            except IndexError as e:
                logger.error("Invalid action %s submitted: %s", str(action), e)
                return self._get_obs(), reward, self.gc.outcome != sts.GameOutcome.UNDECIDED, truncated, self._get_info()

            if action.value in [a.value for a in self.gc.get_available_actions()]:
                if self.act != self.gc.act:
                    logger.debug("Act map: %s", str(self.gc.map))
                    self.act = self.gc.act
                logger.debug("Game context available actions-> %s",
                             ";".join([a.print_desc(self.gc) for a in self.gc.get_available_actions()]))
                logger.debug("Game context execute action-> %s", action.print_desc(self.gc))

                self.gc.execute(action)
                logger.debug("Game context -> %s", str(self.gc))

            # Entering battle?
            if self.is_battle():
                self.bc = sts.BattleContext()
                self.bc.init(self.gc)
                self.turn = -1
        if self.gc.outcome != sts.GameOutcome.UNDECIDED:
            reward = self.gc.get_final_score() / 1000
        elif self.give_continuous_reward:
            reward += self.gc.get_final_score() / 1000
        return self._get_obs(), reward, self.gc.outcome != sts.GameOutcome.UNDECIDED, truncated, self._get_info()

    def render(self) -> str:
        if self.is_battle():
            return str(self.gc) + "\n" + str(self.bc) + "\n\n\n"
        return str(self.gc) + "\n\n\n"


class StsSkipbattleEnvironment(StsEnvironment):
    def set_battle_predictor(self, battle_predictor):
        self._battle_predictor = battle_predictor

    def battle_predict(self, bc):
        return self._battle_predictor.forward(StsEncodings.encode_battle(self.gc, bc))

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        observation, info = super(StsSkipbattleEnvironment, self).reset(seed, options)
        return observation, info

    def step(self, action):
        out = super(StsSkipbattleEnvironment, self).step(action)
        if self.is_battle():
            bc = sts.BattleContext()
            bc.init(self.gc)
            # hp_lost = self.battle_predict(bc)
            while bc.outcome == sts.BattleOutcome.UNDECIDED:
                searcher = sts.BattleAgent(bc)
                searcher.search(1000)
                if len(searcher.best_action_sequence) > 0:
                    bc.execute(searcher.best_action_sequence[0])
                else:
                    bc.execute(bc.get_available_actions()[0])
            bc.exit_battle(self.gc)
            # self.gc.cur_hp -= hp_lost
            if self.gc.outcome != sts.GameOutcome.UNDECIDED:
                reward = self.gc.get_final_score() / 1000
            else:
                reward = 0
            return self._get_obs(), reward, self.gc.outcome != sts.GameOutcome.UNDECIDED, False, self._get_info()
        else:
            return out
