import json
from typing import Optional

import gymnasium
import gymnasium as gym
import numpy as np

from stsrl.game_encoding import StsEncodings
import stsrl.slaythespire as sts


class StsBattleEnvironment(gymnasium.Env):
    def __init__(self):
        super().__init__()
        self.gc = None
        self.bc = None
        self.observation_space = gym.spaces.Box(np.zeros_like(StsEncodings.nniInstance.getBattleObservationMaximums()),
                                                np.array(StsEncodings.nniInstance.getBattleObservationMaximums()))
        self.action_space = gym.spaces.Discrete(StsEncodings.encodingInstance.battle_action_space_size)

    def _get_obs(self):
        return StsEncodings.encode_battle(self.gc, self.bc)

    def _get_info(self):
        return {
            "bc": self.bc,
            "legal_actions": [StsEncodings.encode_battle_action(a) for a in self.bc.get_available_actions()]
        }

    def set_gc(self, gc: sts.GameContext):
        self.gc = gc

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if options is not None and "bc_json" in options:
            self.bc = sts.BattleContext()
            self.bc.init_from_json(self.gc, json.dumps(options["bc_json"]))
        else:
            self.bc = sts.BattleContext()
            self.bc.init(self.gc)

        if seed is not None:
            self.bc.randomize_rng_counters(seed)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        actions = self.bc.get_available_actions()
        action = StsEncodings.decode_battle_action(action)

        # TODO unnecessary
        if action.value in [a.value for a in actions]:
            self.bc.execute(action)
        else:
            raise NotImplementedError(
                f"Somehow we got a invalid action: {action.print_desc(self.bc)}, expected: {[a.print_desc(self.bc) for a in actions]}")

        reward = 0
        truncated = False
        if self.bc.outcome == sts.BattleOutcome.PLAYER_VICTORY:
            reward = 1 + (self.bc.player.hp / self.bc.player.max_hp) + len(self.bc.potions) / 5
        elif self.bc.outcome == sts.BattleOutcome.PLAYER_LOSS:
            reward = -1
        return self._get_obs(), reward, self.bc.outcome != sts.BattleOutcome.UNDECIDED, truncated, self._get_info()


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
            "gc": self.gc,
            "legal_actions": [StsEncodings.encode_game_action(a) for a in self.gc.get_available_actions()],
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is None:
            import random
            seed = random.randint(0, 10000)
        if options is not None:
            self.gc = sts.GameContext()
            sts.init_from_json(json.dumps(options))
        else:
            self.gc = sts.GameContext(sts.CharacterClass.IRONCLAD, 0, seed)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step_out_of_combat(self, bc):
        bc.exit_battle(self.gc)

    def is_battle(self):
        return self.gc.screen_state == sts.ScreenState.BATTLE

    def step(self, action):
        reward = 0
        truncated = False
        if self.gc.screen_state == sts.ScreenState.BATTLE:
            raise NotImplementedError("Game environment cannot execute a battle")

        actions = self.gc.get_available_actions()
        action = StsEncodings.decode_game_action(action)
        # TODO unnecessary
        if action.value in [a.value for a in actions]:
            self.gc.execute(action)
        else:
            raise NotImplementedError(
                f"Somehow we got a invalid action: {action.print_desc(self.gc)}, expected: {[a.print_desc(self.gc) for a in actions]}")

        if self.gc.outcome == sts.GameOutcome.PLAYER_VICTORY:
            reward = 1
        if self.gc.outcome == sts.GameOutcome.PLAYER_LOSS:
            reward = -1
        return self._get_obs(), reward, self.gc.outcome != sts.GameOutcome.UNDECIDED, truncated, self._get_info()


class StsEnvironment(gymnasium.Env):
    def __init__(self):
        self.gc = sts.GameContext(sts.CharacterClass.IRONCLAD, 0, 0)
        self.bc = None

        self.observation_space = gym.spaces.Dict(
            {
                # There is probably a better way to describe the environment
                "game": gym.spaces.Box(np.zeros_like(StsEncodings.nniInstance.getObservationMaximums()),
                                       np.array(StsEncodings.nniInstance.getObservationMaximums())),
                "battle": gym.spaces.Box(np.zeros_like(StsEncodings.nniInstance.getBattleObservationMaximums()),
                                         np.array(StsEncodings.nniInstance.getBattleObservationMaximums())),
            }
        )
        self.action_space = gym.spaces.Discrete(StsEncodings.encodingInstance.battle_action_space_size)

    def _get_obs(self):
        if self.gc.screen_state == sts.ScreenState.BATTLE:
            return {
                "game": StsEncodings.encode_game(self.gc),
                "battle": StsEncodings.encode_battle(self.gc, self.bc)
            }
        else:
            return {
                "game": StsEncodings.encode_game(self.gc),
                "battle": np.zeros(StsEncodings.nniInstance.battle_space_size, dtype=np.float32)
            }

    def _get_info(self):
        return {
            "gc": self.gc,
            "bc": self.bc,
            "legal_actions": [StsEncodings.encode_battle_action(a) for a in self.bc.get_available_actions()] if self.is_battle() else [StsEncodings.encode_battle_action(a) for a in self.gc.get_available_actions()]
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super(StsEnvironment, self).reset(seed=seed)
        if seed is None:
            import random
            seed = random.randint(0, 10000)
        if options is not None:
            self.gc = sts.GameContext()
            sts.init_from_json(json.dumps(options))
        else:
            self.gc = sts.GameContext(sts.CharacterClass.IRONCLAD, 0, seed)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def _battle_step(self, action):
        action = StsEncodings.decode_battle_action(action)
        self.bc.execute(action)

    def is_battle(self):
        return self.gc.screen_state == sts.ScreenState.BATTLE

    def step(self, action):
        reward = 0
        truncated = False
        if self.gc.screen_state == sts.ScreenState.BATTLE:
            self._battle_step(action)
            if self.bc.is_terminal():
                self.bc.exit_battle(self.gc)
        else:
            action = StsEncodings.decode_game_action(action)
            self.gc.execute(action)
            # Entering battle?
            if self.gc.screen_state == sts.ScreenState.BATTLE:
                self.bc = sts.BattleContext()
                self.bc.init(self.gc)
        if self.gc.outcome == sts.GameOutcome.PLAYER_VICTORY:
            reward = 1
        if self.gc.outcome == sts.GameOutcome.PLAYER_LOSS:
            reward = -1
        return self._get_obs(), reward, self.gc.outcome != sts.GameOutcome.UNDECIDED, truncated, self._get_info()
