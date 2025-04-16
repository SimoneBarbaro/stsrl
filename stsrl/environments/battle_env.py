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
        self.observation_space = gym.spaces.Dict(
            {
                "observations": gym.spaces.Box(np.zeros_like(StsEncodings.nniInstance.getBattleObservationMaximums()),
                                               np.ones_like(StsEncodings.nniInstance.getBattleObservationMaximums())),
                "action_mask": gym.spaces.MultiBinary(StsEncodings.encodingInstance.battle_action_space_size),
            }
        )

        self.action_space = gym.spaces.Discrete(StsEncodings.encodingInstance.battle_action_space_size)
        self.turn = 0
        self.starting_hp = 0
        self.prev_hp = 0

    def _get_valid_action_mask(self):
        if self.bc.outcome != sts.GameOutcome.UNDECIDED:
            return np.zeros(StsEncodings.encodingInstance.battle_action_space_size, dtype=bool)

        return np.array(self.bc.get_valid_actions_mask(), dtype=bool)

    def _get_obs(self):
        return {
            "observations": np.array(StsEncodings.encode_battle(
                self.gc,
                self.bc) / StsEncodings.nniInstance.getBattleObservationMaximums(), dtype=np.float32),
            "action_mask": np.array(self.bc.get_valid_actions_mask(),
                                    dtype=bool) if self.bc.outcome == sts.BattleOutcome.UNDECIDED else np.zeros(
                StsEncodings.encodingInstance.battle_action_space_size, dtype=bool)
        }

    def _get_info(self):
        return {
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
            which_battle = random.randint(0, 60)
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

        if self.bc.outcome != sts.BattleOutcome.UNDECIDED:
            return self.reset()

        logger.debug("Game context Reset -> %s", str(self.gc))
        logger.debug("Battle context Reset -> %s", str(self.bc))

        if seed is not None:
            self.bc.randomize_rng_counters(seed)

        observation = self._get_obs()
        info = self._get_info()
        self._reset_internal_variables()

        return observation, info

    def _reset_internal_variables(self):
        self.turn = -1
        self.starting_hp = self.bc.player.hp
        self.prev_hp = self.starting_hp

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
        if self.bc.is_terminal():
            logger.debug("Battle context end state -> %s", str(self.bc))
            if self.bc.outcome == sts.BattleOutcome.PLAYER_LOSS:
                reward = -1
            else:
                reward = (self.bc.player.hp / self.starting_hp) + (self.bc.player.max_hp / self.gc.max_hp - 1)
            terminated = True
        # Let's try some reward for progressing battle
        elif self.turn != self.bc.turn:
            reward += (self.bc.player.hp - self.prev_hp) / self.bc.player.max_hp
            self.prev_hp = self.bc.player.hp
            for m in self.bc.monsters:
                if m.max_hp > 0:
                    reward += 0.2 * (m.max_hp - m.hp) / m.max_hp / len(self.bc.monsters)
        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self) -> str:
        return str(self.bc)


class StsBattleFromSavesEnvironment(StsBattleEnvironment):
    """Environment that plays through battles given at init."""

    def __init__(self, battles):
        """
        :param battles: List of battles, each battle must be a json string representing the battle in the format of the communication mod
        """
        super().__init__()
        self.battles = battles

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            random.seed(seed)
        game_string = random.choice(self.battles)
        self.gc = sts.GameContext()
        self.gc.init_from_json(game_string)
        self.bc = sts.BattleContext()
        self.bc.init_from_json(self.gc, game_string)

        logger.debug("Game context Reset -> %s", str(self.gc))
        logger.debug("Battle context Reset -> %s", str(self.bc))

        # TODO a bit of randomization of the deck/relics here?

        observation = self._get_obs()
        info = self._get_info()
        self._reset_internal_variables()

        return observation, info
