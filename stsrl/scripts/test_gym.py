import logging
import os

import gymnasium as gym
from skrl.agents.torch.a2c import A2C_DEFAULT_CONFIG
# import the skrl components to build the RL system
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.utils import set_seed

from stsrl.gym.agents import StsGameAgent
from stsrl.gym.models import ActorMLP, CriticMLP
from stsrl.gym.trainer import StsTrainer

logger = logging.getLogger(__name__)
dir_path = os.path.dirname(os.path.realpath(__file__))
logging.basicConfig(filename=os.path.join(dir_path, "../../logs", f"test-gym.log"),
                    level=logging.DEBUG)


def get_agent(env, agent_name):
    models = {}
    models["policy"] = ActorMLP(observation_space=env.observation_space,
                                action_space=env.action_space,
                                device=env.device,
                                unnormalized_log_prob=True)
    models["value"] = CriticMLP(observation_space=env.observation_space,
                                action_space=env.action_space,
                                device=env.device,
                                clip_actions=False)

    cfg = A2C_DEFAULT_CONFIG.copy()
    cfg["learning_starts"] = 5000
    cfg["experiment"]["write_interval"] = 1000
    cfg["experiment"]["checkpoint_interval"] = 5000
    cfg["experiment"]["directory"] = os.path.join(dir_path, "runs/test", agent_name)
    cfg["experiment"]["experiment_name"] = "test-a2c-mlp"

    # instantiate the agent
    # (assuming a defined environment <env> and memory <memory>)

    class MyMemory(RandomMemory):
        def create_tensor(self, name, size, dtype=None, keep_dimensions=False):
            return super().create_tensor(name, size, dtype, keep_dimensions)

    # instantiate a memory as experience replay
    memory = MyMemory(memory_size=cfg["learning_starts"], num_envs=env.num_envs, device=env.device, replacement=False)

    # initialize models' parameters (weights and biases)
    for model in models.values():
        model.init_parameters(method_name="normal_", mean=0.0, std=0.1)

    # return DebugAgent(actionSequenceFile=os.path.join(dir_path, "logs", "action-sequence.log"),
    return StsGameAgent(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device)


# seed for reproducibility
set_seed(42)  # e.g. `set_seed(42)` for fixed seed

env = wrap_env(gym.make('sts-game'))
battle_env = wrap_env(gym.make('sts-battle'))

env._device = "cpu"
battle_env._device = "cpu"

# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 10000, "headless": True, "environment_info": "environment_info"}
# trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=[agent])
trainer = StsTrainer(
    cfg=cfg_trainer, env=env, battle_env=battle_env, agent=get_agent(env, "game_a2c"),
    battle_agent=get_agent(battle_env, "battle_a2c"))

# start training
trainer.train()
