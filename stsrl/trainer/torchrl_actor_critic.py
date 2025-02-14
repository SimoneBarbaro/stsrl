import collections
import datetime
import glob
import logging
import os
import random

import numpy as np
import torch
from tensordict.nn import TensorDictModule, InteractionType
from torchrl.collectors import SyncDataCollector
from torchrl.data import ReplayBuffer, LazyTensorStorage, SamplerWithoutReplacement
from torchrl.modules import ProbabilisticActor, MaskedCategorical, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.record.loggers.tensorboard import TensorboardLogger
from tqdm import tqdm

from stsrl.environments.env_factory import make_torchrl_env
from stsrl.models.model_factory import build_model

logger = logging.getLogger(__name__)


class TorchPPOExperiment:
    def __init__(self, config):
        self.config = config
        self.seed = config["seed"]
        self.device = config["device"]
        self.modules = {}
        self.game_actor, self.game_critic = None, None
        self.battle_actor, self.battle_critic = None, None
        self.battle_replay_buffer, self.game_replay_buffer = None, None
        self.experiment_dir = config["experiment_dir"]
        os.makedirs(os.path.join(self.experiment_dir, "checkpoints"), exist_ok=True)

        torch.manual_seed(self.seed)
        random.seed(self.seed)

    def _get_actor_critic(self, policy_module, value_module, action_spec):
        actor = ProbabilisticActor(
            module=policy_module,
            spec=action_spec,
            in_keys=["logits", "mask"],
            distribution_class=MaskedCategorical,
            distribution_kwargs={
            },
            return_log_prob=True,
            default_interaction_type=InteractionType.RANDOM
            # we'll need the log-prob for the numerator of the importance weights
        )

        return actor, value_module

    def init_models(self):
        env = make_torchrl_env("sts", "cpu")

        policy_model = build_model(output_len=env.action_spec.n, softmax_head=True, **self.config["model"])
        value_model = build_model(output_len=1, softmax_head=False, **self.config["model"])
        self.game_actor, self.game_critic = self._get_actor_critic(
            TensorDictModule(
                module=policy_model,
                in_keys=["game", "battle", "is_battle"], out_keys=["logits"]
            ),
            ValueOperator(
                module=value_model,
                in_keys=["game", "battle", "is_battle"],
            ), env.action_spec
        )
        self.battle_actor, self.battle_critic = self._get_actor_critic(
            TensorDictModule(
                module=policy_model.battle_model,
                in_keys=["battle"], out_keys=["logits"]
            ),
            ValueOperator(
                module=value_model.battle_model,
                in_keys=["battle"],
            ), env.action_spec
        )

        self.battle_replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(max_size=self.config["replay_size"]),
            sampler=SamplerWithoutReplacement(),
        )

        self.game_replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(max_size=self.config["replay_size"]),
            sampler=SamplerWithoutReplacement(),
        )

        self.modules = {
            "policy": policy_model,
            "critic": value_model,
            "battle_replay": self.battle_replay_buffer,
            "game_replay": self.game_replay_buffer
        }

    def _save_models(self, tag=None, only_latest=True):
        for f in glob.glob(os.path.join(
                os.path.join(self.experiment_dir, "checkpoints"),
                f"modules_{'_'.join(tag.split('_')[:-1])}*")):
            os.remove(f)
        if tag is None:
            tag = str(datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S-%f"))
        torch.save(self.modules, os.path.join(self.experiment_dir, "checkpoints", f"modules_{tag}.pt"))

    def _load_models(self, path=None):
        if path is None:
            path = os.path.join(self.experiment_dir, "checkpoints", f"models_latest.pt")
        modules = torch.load(path)
        for name, module in self.modules.items():
            module.load_state_dict(modules[name])

    def _train(self, env, actor, critic, replay_buffer, train_config):
        tb_logger = TensorboardLogger(train_config["experiment_name"], log_dir=self.experiment_dir)
        actor.to(self.config["device"])
        critic.to(self.config["device"])
        advantage_module = GAE(
            gamma=self.config["gamma"],
            lmbda=self.config["lmbda"],
            value_network=critic, average_gae=True
        )
        loss_module = ClipPPOLoss(
            actor_network=actor,
            critic_network=critic,
            entropy_bonus=bool(train_config["entropy_eps"]),
            entropy_coef=train_config["entropy_eps"],
            critic_coef=1.0,
            loss_critic_type="smooth_l1",
        )
        optim = torch.optim.Adam(loss_module.parameters(), train_config["learning_rate"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, train_config["annealing_max_T"], 0.0
        )
        self.modules["optim"] = optim
        self.modules["scheduler"] = scheduler
        timestep = 0

        try:
            latest_file = max(glob.glob(os.path.join(
                os.path.join(self.experiment_dir, "checkpoints"),
                f"modules_{train_config['experiment_name']}*")), key=os.path.getctime)
            if latest_file:
                self._load_models(latest_file)
                timestep = int(latest_file.split("_")[-1].split(".")[0])
        except ValueError:
            logger.info("No checkpoint found")

        collector = SyncDataCollector(
            make_torchrl_env(env, device=self.device),
            actor,
            split_trajs=False,
            device=self.device,
            frames_per_batch=train_config["batch_size"],
            total_frames=train_config["num_iterations"]*train_config["batch_size"] - timestep
        )
        # Collect data and track cumulative rewards
        tracking_data = collections.defaultdict(list)
        pbar = tqdm(total=train_config["num_iterations"]*train_config["batch_size"], initial=timestep)

        for i, tensordict_data in enumerate(collector):
            timestep += tensordict_data.numel()
            advantage_module(tensordict_data)
            data_view = tensordict_data.reshape(-1)
            replay_buffer.extend(data_view.cpu())

            tracking_data["episode_len"].append(torch.mean(data_view["step_count"].float()).item())
            if "reward" in data_view["next"]:
                tracking_data["reward"].append(torch.mean(data_view["next"]["reward"]).item())
                tracking_data["episode_reward"].append(torch.mean(
                    data_view["next"]["episode_reward"][data_view["next"]["terminated"]]).item())

            for _ in range(train_config["num_epochs"]):
                subdata = replay_buffer.sample(train_config["batch_size"])
                loss_vals = loss_module(subdata.to(self.device))
                loss_value = (
                        loss_vals["loss_objective"]
                        + loss_vals["loss_critic"]
                        + loss_vals["loss_entropy"]
                )

                tracking_data["loss_objective"].append(torch.mean(loss_vals["loss_objective"]).item())
                tracking_data["loss_critic"].append(torch.mean(loss_vals["loss_critic"]).item())
                tracking_data["loss_entropy"].append(torch.mean(loss_vals["loss_entropy"]).item())

                # Optimization: backward, grad clipping and optimization step
                loss_value.backward()
                # this is not strictly mandatory but it's good practice to keep
                # your gradient norm bounded
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), 1.)
                optim.step()
                optim.zero_grad()
            if i % self.config["log_interval"] == 0:
                tb_logger.log_scalar("Episode length (mean)", np.mean(tracking_data["episode_len"]).item())
                tb_logger.log_scalar("Loss (objective)", np.mean(tracking_data["loss_objective"]).item(), timestep)
                tb_logger.log_scalar("Loss (critic)", np.mean(tracking_data["loss_critic"]).item(), timestep)
                tb_logger.log_scalar("Loss (entropy)", np.mean(tracking_data["loss_entropy"]).item(), timestep)
                tb_logger.log_scalar("Learning rate", scheduler.get_lr()[0], timestep)

                tb_logger.log_scalar("Reward (mean)", np.mean(tracking_data["reward"]).item(), timestep)
                tb_logger.log_scalar("Episode Reward (mean)", np.mean(tracking_data["episode_reward"]).item(), timestep)

                tracking_data["episode_len"].clear()
                tracking_data["reward"].clear()
                tracking_data["episode_reward"].clear()
                tracking_data["loss_objective"].clear()
                tracking_data["loss_critic"].clear()
                tracking_data["loss_entropy"].clear()

            pbar.update(tensordict_data.numel())
            if i % self.config["checkpoint_interval"] == 0:
                self._save_models(tag=train_config["experiment_name"]+"_"+str(timestep))
            scheduler.step()

    def train_battle(self):
        self._train("sts-battle", self.battle_actor, self.battle_critic,
                    self.battle_replay_buffer, self.config["battle_train_config"])

    def train_game(self):
        self._train("sts", self.battle_actor, self.battle_critic,
                    self.game_replay_buffer, self.config["game_train_config"])
