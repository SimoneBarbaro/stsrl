import argparse
import logging
import os
import time

import gymnasium as gym
import ray
from ray import tune, train
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.core.models.configs import ModelConfig
from ray.rllib.core.rl_module import RLModuleSpec
from ray.rllib.examples.rl_modules.classes.action_masking_rlm import ActionMaskingTorchRLModule

from stsrl.environments.battle_env import StsBattleFromSavesEnvironment
from stsrl.models.game_embeddings import StsEmbeddingModuleEncoderConfig


logger = logging.getLogger(__name__)
dir_path = os.path.dirname(os.path.realpath(__file__))
logging.basicConfig(#filename=os.path.join(dir_path, "logs", f"ray_train_battle_model.log"),
                    level=logging.DEBUG,
                    filemode="w")


def _env_creator(cfg):
    return StsBattleFromSavesEnvironment(**cfg)


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


def main(args):
    with open(args.battle_file) as f:
        battles = f.readlines()

    ray.init()

    tune.register_env('stsBattle', _env_creator)

    config = (
        PPOConfig().resources(
            num_gpus=1
        ).api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        ).framework(
            framework="torch",
        ).environment(
            "stsBattle",
            action_mask_key="action_mask",
            env_config={
                "battles": battles,
            },

        ).rl_module(
            model_config={
                "custom_encoding": True,
                "vf_share_layers": True,
                "head_fcnet_hiddens": tune.grid_search([[128 for l in range(n)] for n in [4, 8, 16]]),
                "head_fcnet_activation": "tanh",
            },
            rl_module_spec=RLModuleSpec(
                module_class=ActionMaskingTorchRLModule,
                catalog_class=BattleCatalog
            ),
        ).training(
            use_critic=True,
            use_gae=True,
            use_kl_loss=True,
            lr=0.01,
        ).env_runners(
            num_env_runners=1
        ).debugging(
            log_level="DEBUG",
        ).fault_tolerance(
            # Recreate any failed EnvRunners.
            restart_failed_env_runners=True,
        )
    )

    path = os.path.join(dir_path, "experiments", "ray_tune")
    experiment_name = "ppo_battle"
    experiment_dir = os.path.join(path, experiment_name)
    if not tune.Tuner.can_restore(experiment_dir):
        tuner = tune.Tuner(
            "PPO",
            param_space=config,
            run_config=train.RunConfig(
                stop={"env_runners/episode_return_mean": 1.0},
                storage_path=path, name=experiment_name,
                checkpoint_config=train.CheckpointConfig(
                    checkpoint_frequency=10, checkpoint_at_end=True
                ),
                failure_config=train.FailureConfig(max_failures=-1)
            ),
        )
        tuner.fit()
    else:
        tuner = tune.Tuner.restore(
            experiment_dir, "PPO",
            param_space=config,
            restart_errored=True,
        )
        tuner.fit()
    algo = Algorithm.from_checkpoint(
        tuner.get_results()
        .get_best_result(
            metric="env_runners/episode_return_mean", mode="max")
        .checkpoint.path)
    ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='RayTrainBattleModel',
        description='Runs a ray tune job on the stored battles passed in the file given as argument',
    )
    parser.add_argument('--battle-file', help="file where the battles are stored", type=str)
    args = parser.parse_args()

    main(args)
