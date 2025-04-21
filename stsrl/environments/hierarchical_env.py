import logging
from typing import Optional

import gymnasium as gym
import numpy as np
import stsrl.slaythespire as sts
from ray.rllib import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict

from stsrl.environments import StsEnvironment

logger = logging.getLogger(__name__)


class HierarchicalStsEnvironment(StsEnvironment, MultiAgentEnv):
    def __init__(self, **kwargs):
        StsEnvironment.__init__(self, **kwargs)
        MultiAgentEnv.__init__(self)
        self._game_agent_id = "game"
        self.observation_spaces = {
            self._game_agent_id: gym.spaces.Dict({
                "observations": self.observation_space[self._game_agent_id],
                "action_mask": self.observation_space["legal_actions_mask"]
            }),
            "battle": gym.spaces.Dict({
                "observations": self.observation_space["battle"],
                "action_mask": self.observation_space["legal_actions_mask"]
            }),
        }

        self.observation_spaces.update(
            {
                f"battle_{i}": self.observation_spaces["battle"]
                for i in range(100)
            }
        )
        self.action_spaces = {
            self._game_agent_id: self.action_space,
            #"battle": self.action_space
        }

        self.action_spaces.update(
            {
                f"battle_{i}": self.action_space
                for i in range(100)
            }
        )

        self.observation_space = None  # gym.spaces.Dict(self.observation_spaces)
        self.action_space = None  # gym.spaces.Dict(self.action_spaces)
        self.agents = self.possible_agents = [self._game_agent_id] + [ f"battle_{i}" for i in range(100)]
        self._battle_agent_prefix = "battle_"
        self._battle_count = -1  # Will be incremented when a battle starts
        self._current_battle_agent_id = None  # Track the ID of the *current* battle instance

    # --- Helper to determine active agent ID ---
    def _get_current_agent_id(self) -> str:
        """Returns the ID of the agent currently controlling the environment."""
        if self.is_battle():
            # If we are in a battle but don't have a current battle ID assigned yet,
            # it means a new battle just started.
            if self._current_battle_agent_id is None:
                self._battle_count += 1
                self._current_battle_agent_id = f"{self._battle_agent_prefix}{self._battle_count}"

                self.observation_spaces[self._current_battle_agent_id] = self.observation_spaces[self._current_battle_agent_id]
                self.action_spaces[self._current_battle_agent_id] = self.action_spaces[self._current_battle_agent_id]

            return self._current_battle_agent_id
        else:
            # If we are not in battle, the game agent is active.
            # Reset the current battle ID tracker.
            self._current_battle_agent_id = None
            return self._game_agent_id

    # --- Override _get_info ---
    def _get_info(self) -> MultiAgentDict:
        """Returns info dictionary keyed by the active agent's specific ID."""
        infos = super()._get_info()  # Get base info
        active_agent_id = self._get_current_agent_id()  # Get specific ID (game or battle_N)
        return {active_agent_id: infos}

        specific_info = {}
        # Check if the active agent is a battle agent or the game agent
        if active_agent_id.startswith(self._battle_agent_prefix):
            specific_info = {
                "bc": sts.BattleContext(self.bc),  # Assuming self.bc exists
                "gc": sts.GameContext(self.gc),
                **infos
            }
        elif active_agent_id == self._game_agent_id:
            specific_info = {
                "gc": sts.GameContext(self.gc),
                **infos
            }
        # Return dictionary keyed by the *specific* active agent ID
        return {active_agent_id: specific_info}

    # --- Override _get_obs ---
    def _get_obs(self) -> MultiAgentDict:
        """Returns observation dictionary keyed by the active agent's specific ID."""
        obs = super()._get_obs()  # Get base observation structure
        active_agent_id = self._get_current_agent_id()  # Get specific ID (game or battle_N)

        # Check if the active agent is a battle agent or the game agent
        if active_agent_id.startswith(self._battle_agent_prefix):
            # Ensure the underlying obs keys exist
            if "battle" in obs and "legal_actions_mask" in obs:
                return {
                    active_agent_id: {  # Use the specific battle ID (battle_N)
                        "observations": obs["battle"].astype(np.float32),
                        "action_mask": obs["legal_actions_mask"].astype(np.bool_)
                    }
                }
            else:
                logger.error(f"Warning: Missing expected keys in base observation during battle state: {obs.keys()}")
                return {}  # Handle gracefully

        elif active_agent_id == self._game_agent_id:
            if "game" in obs and "legal_actions_mask" in obs:
                return {
                    self._game_agent_id: {  # Use the game ID
                        "observations": obs["game"].astype(np.float32),
                        "action_mask": obs["legal_actions_mask"].astype(np.bool_)
                    }
                }
            else:
                logger.error(f"Warning: Missing expected keys in base observation during game state: {obs.keys()}")
                return {}  # Handle gracefully
        else:
            logger.error(f"THIS SHOULD NOT HAPPEN WHY ARE WE EVEN HERE TODO REMOVE THIS BRANCH")
            return {}  # Handle gracefully

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self._battle_count = -1
        self._current_battle_agent_id = None

        return super().reset(seed=seed, options=options)

    def step(self, action_dict):
        # Identify the agent ID that *was* active and provided the action
        # action_dict should contain exactly one key: the ID of the agent who needs to act
        if len(action_dict) != 1:
            raise ValueError(f"Expected action_dict with exactly one agent ID, but got {action_dict.keys()}")
        agent_that_acted = list(action_dict.keys())[0]
        was_battle = self.is_battle()
        if was_battle:
            assert (self._current_battle_agent_id == agent_that_acted)

        obs, reward, terminated, truncated, info = super(HierarchicalStsEnvironment, self).step(
            action_dict[agent_that_acted])
        # Let's assign reward only to the agent that acted.
        rewards = {agent_that_acted: reward}
        # --- Termination Logic ---
        terminateds = {}
        battle_just_ended = was_battle and not self.is_battle()
        if battle_just_ended:
            if self.bc.outcome:
                rewards["agent_that_acted"] = -1
            terminateds[agent_that_acted] = True
        if terminated:
            rewards["game"] = reward
            rewards[agent_that_acted] = -1 if self.bc.outcome == sts.BattleOutcome.PLAYER_LOSS else 1

        terminateds = {"__all__": terminated}
        truncateds = {"__all__": truncated}
        return obs, rewards, terminateds, truncateds, info
