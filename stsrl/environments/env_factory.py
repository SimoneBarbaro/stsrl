import torch
import gymnasium as gym
from torchrl.envs import GymWrapper, TransformedEnv, Compose, StepCounter, DTypeCastTransform, RewardSum


def make_torchrl_env(env_type="sts", device="cuda"):
    base_env = GymWrapper(gym.make(env_type), device=device, categorical_action_encoding=True)

    return TransformedEnv(
        base_env,
        Compose(
            StepCounter(),
            RewardSum(),
            DTypeCastTransform(
                in_keys=["legal_actions_mask"], out_keys=["mask"],
                in_keys_inv=[],
                dtype_in=torch.float32, dtype_out=torch.bool),
        ), )
