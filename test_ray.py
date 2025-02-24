import logging
import os

import gymnasium as gym
import torch.nn
from ray import tune, train
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.core.models.base import Encoder, ENCODER_OUT
from ray.rllib.core.models.configs import ModelConfig
from ray.rllib.core.models.torch.base import TorchModel
from ray.rllib.core.rl_module import RLModuleSpec, MultiRLModuleSpec, MultiRLModule
from ray.rllib.examples.rl_modules.classes.action_masking_rlm import ActionMaskingTorchRLModule

from stsrl.environments.ray import HierarchicalStsEnvironment
from stsrl.models.game_embeddings import BattleEmbeddingModule, GameEmbeddingModule

logger = logging.getLogger(__name__)
dir_path = os.path.dirname(os.path.realpath(__file__))
logging.basicConfig(filename=os.path.join(dir_path, "logs", f"test-ray.log"),
                    level=logging.DEBUG,
                    filemode="w", )


def _env_creator(cfg):
    return HierarchicalStsEnvironment(**cfg)


def policy_mapping_fn(agent_id, episode, **kwargs):
    # Map each low level agent to its respective (low-level) policy.
    if agent_id.startswith("battle"):
        return "battle_policy"
    # Map the high level agent to the high level policy.
    else:
        return "game_policy"


class StsEmbeddingModuleEncoderConfig(ModelConfig):
    output_dims = (256,)
    freeze = False

    def __init__(self, encode_battle=False, output_dim=256, num_layers=1):
        super().__init__()
        self.encode_battle = encode_battle
        self.output_dims = (output_dim,)
        self.num_layers = num_layers

    def build(self, framework):
        assert framework == "torch", "Unsupported framework `{}`!".format(framework)
        return StsEmbeddingModuleEncoder(self)


class StsEmbeddingModuleEncoder(TorchModel, Encoder):
    def __init__(self, config: StsEmbeddingModuleEncoderConfig):
        TorchModel.__init__(self, config)
        Encoder.__init__(self, config)
        if config.encode_battle:
            embedding = BattleEmbeddingModule()
        else:
            embedding = GameEmbeddingModule()
        self.net = torch.nn.Sequential(
            embedding,
            torch.nn.ReLU(),
        )
        self.net.append(
            torch.nn.Linear(embedding.embedding_size, config.output_dims[0]))
        self.net.append(
            torch.nn.ReLU())
        for l in range(1, config.num_layers):
            self.net.append(
                torch.nn.Linear(config.output_dims[0], config.output_dims[0]))
            self.net.append(
                torch.nn.ReLU())

    def _forward(self, input_dict: dict, **kwargs) -> dict:
        return {ENCODER_OUT: (self.net(input_dict["obs"]))}


class BattleCatalog(PPOCatalog):
    @classmethod
    def _get_encoder_config(
        cls,
        observation_space: gym.Space,
        model_config_dict: dict,
        action_space: gym.Space = None,
    ) -> ModelConfig:
        if model_config_dict.get("custom_encoding", "False"):
            return StsEmbeddingModuleEncoderConfig(
                encode_battle=True,
                num_layers=model_config_dict.get("battle_encoder_num_layers", 1),
                output_dim=model_config_dict.get("battle_encoder_output_dim", 256))
        super()._get_encoder_config(observation_space, model_config_dict, action_space)


class GameCatalog(PPOCatalog):
    @classmethod
    def _get_encoder_config(
        cls,
        observation_space: gym.Space,
        model_config_dict: dict,
        action_space: gym.Space = None,
    ) -> ModelConfig:
        if model_config_dict.get("custom_encoding", "False"):
            return StsEmbeddingModuleEncoderConfig(
                encode_battle=False,
                num_layers=model_config_dict.get("game_encoder_num_layers", 1),
                output_dim=model_config_dict.get("game_encoder_output_dim", 256))
        super()._get_encoder_config(observation_space, model_config_dict, action_space)


def main():

    tune.register_env('sts', _env_creator)

    config = (
        PPOConfig()
        .resources(
            num_gpus=1
        )
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .framework(
            framework="torch",
        )
        .environment(
            "sts",
            action_mask_key="legal_actions_mask"
        )
        .rl_module(
            model_config={
                "custom_encoding": tune.grid_search([True,
                                                     #False
                                                     ]),
                #"custom_encoding": tune.grid_search([True, False]),
                #'battle_encoder_num_layers': tune.grid_search([1, 2]),
                #'game_encoder_num_layers': tune.grid_search([1, 2]),
                "vf_share_layers": tune.grid_search([True, False]),
                "head_fcnet_hiddens":
                    [128 for l in range(4)],
                    # tune.grid_search([[128 for l in range(n)] for n in [4]]),
                "head_fcnet_activation": tune.choice(["relu", "tanh"]),
            },
            # We need to explicitly specify here RLModule to use and
            # the catalog needed to build it.
            rl_module_spec=MultiRLModuleSpec(
                multi_rl_module_class=MultiRLModule,
                rl_module_specs={
                    "battle_policy": RLModuleSpec(
                        module_class=ActionMaskingTorchRLModule,
                        catalog_class=BattleCatalog

                    ),
                    "game_policy": RLModuleSpec(
                        module_class=ActionMaskingTorchRLModule,
                        catalog_class=GameCatalog
                    )
                }
            ),
        ).training(
            use_critic=True,
            use_gae=True,
            use_kl_loss=True,  # tune.grid_search([True, False]),
            lr=0.01,
        ).env_runners(num_env_runners=1)
        .debugging(
            log_level="DEBUG",
        ).multi_agent(
            policy_mapping_fn=policy_mapping_fn,
            policies={"game_policy", "battle_policy"},
        )
    )
    path = os.path.join(dir_path, "experiments", "ray_tune")

    experiment_dir = os.path.join(path, "ppo_test_encoder")
    if not tune.Tuner.can_restore(experiment_dir):
        tuner = tune.Tuner(
            "PPO",
            param_space=config,
            run_config=train.RunConfig(
                stop={"ray/tune/env_runners/agent_episode_returns_mean/game": 1.0},
                storage_path=path, name="ppo_test_encoder",
                checkpoint_config=train.CheckpointConfig(
                    checkpoint_frequency=10, checkpoint_at_end=True
                ),
            ),
        )
        tuner.fit()
    else:
        tuner = tune.Tuner.restore(experiment_dir, "PPO", param_space=config, restart_errored=True)
    algo = Algorithm.from_checkpoint(
        tuner.get_results()
        .get_best_result(
            metric="env_runners/episode_return_mean", mode="max")
        .checkpoint.path)
    print(algo.get_module("game_policy"))


if __name__ == "__main__":
    main()
