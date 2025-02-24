import gymnasium as gym
from ray.rllib import MultiAgentEnv

from stsrl.environments import StsEnvironment


class HierarchicalStsEnvironment(StsEnvironment, MultiAgentEnv):
    def __init__(self):
        StsEnvironment.__init__(self)
        MultiAgentEnv.__init__(self)
        self.observation_spaces = {
            "game": gym.spaces.Dict({
                "observations": self.observation_space["game"],
                "action_mask": self.observation_space["legal_actions_mask"]
            }),
            "battle": gym.spaces.Dict({
                "observations": self.observation_space["battle"],
                "action_mask": self.observation_space["legal_actions_mask"]
            }),
        }
        self.action_spaces = {
            "game": self.action_space,
            "battle": self.action_space
        }
        self.observation_space = None  # gym.spaces.Dict(self.observation_spaces)
        self.action_space = None  # gym.spaces.Dict(self.action_spaces)
        self.agents = self.possible_agents = ["game", "battle"]

    def _get_obs(self):
        obs = super()._get_obs()
        if self.is_battle():
            return {
                "battle": {
                    "observations": obs["battle"].astype("float32"),
                    "action_mask": obs["legal_actions_mask"].astype("bool")
                }
            }
        return {
            "game": {
                "observations": obs["game"].astype("float32"),
                "action_mask": obs["legal_actions_mask"].astype("bool")
            }
        }

    def step(self, action_dict):
        rewards = {"game": 0, "battle": 0}

        if self.is_battle():
            obs, reward, terminated, truncated, info = super(HierarchicalStsEnvironment, self).step(
                action_dict["battle"])
            rewards["battle"] = reward
        else:
            obs, reward, terminated, truncated, info = super(HierarchicalStsEnvironment, self).step(action_dict["game"])
            rewards["game"] = reward

        terminateds = {"__all__": terminated}
        truncateds = {"__all__": truncated}
        return obs, rewards, terminateds, truncateds, info

