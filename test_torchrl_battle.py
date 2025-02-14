import logging
import os
import random
from collections import defaultdict

import torch
from torch import nn

from tensordict.nn import TensorDictModule, InteractionType
from torchrl.collectors import SyncDataCollector
from torchrl.data import ReplayBuffer, LazyTensorStorage, SamplerWithoutReplacement
from torchrl.envs.utils import set_exploration_type
from torchrl.modules import ProbabilisticActor, MaskedCategorical, ValueOperator
from torchrl.objectives import A2CLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm

from stsrl.environments.env_factory import make_torchrl_env
from stsrl.models.game_embeddings import BattleEmbeddingModule
from stsrl.models.sts_mlp import SimpleMlpModule

torch.manual_seed(42)
random.seed(42)

logger = logging.getLogger(__name__)
dir_path = os.path.dirname(os.path.realpath(__file__))
logging.basicConfig(filename=os.path.join(dir_path, "logs", f"test-torchrl-battle.log"),
                    level=logging.DEBUG,
                    filemode="w",
                    )

device = "cuda"

env = make_torchrl_env(env_type="sts-battle")
env.set_seed(42)

print("observation_spec:", env.observation_spec)
print("reward_spec:", env.reward_spec)
print("input_spec:", env.input_spec)
print("action_spec (as defined by input_spec):", env.action_spec)

num_cells = 256  # number of cells in each layer i.e. output dim.
lr = 3e-4
max_grad_norm = 1.0

sub_batch_size = 64  # cardinality of the sub-samples gathered from the current data in the inner loop
num_epochs = 10  # optimisation steps per batch of data collected
clip_epsilon = (
    0.2  # clip value for PPO loss: see the equation in the intro for more context.
)
gamma = 0.99
lmbda = 0.95
entropy_eps = 1e-4
battle_embedding = BattleEmbeddingModule()
policy_module = TensorDictModule(
    nn.Sequential(
        battle_embedding,
        nn.Linear(battle_embedding.embedding_size, num_cells),
        nn.Tanh(),
        nn.Linear(num_cells, num_cells),
        nn.Tanh(),
        nn.Linear(num_cells, num_cells),
        nn.Tanh(),
        nn.Linear(num_cells, env.action_spec.n),
        nn.Softmax()),
    in_keys=["battle"], out_keys=["logits"]
)
policy_module.to(device)
policy_module = ProbabilisticActor(
    module=policy_module,
    spec=env.action_spec,
    in_keys=["logits", "mask"],
    distribution_class=MaskedCategorical,
    distribution_kwargs={
    },
    return_log_prob=True,
    default_interaction_type=InteractionType.RANDOM
    # we'll need the log-prob for the numerator of the importance weights
)

value_module = ValueOperator(
    module=nn.Sequential(
        battle_embedding,
        nn.Linear(battle_embedding.embedding_size, num_cells),
        nn.Tanh(),
        nn.Linear(num_cells, num_cells),
        nn.Tanh(),
        nn.Linear(num_cells, num_cells),
        nn.Tanh(),
        nn.Linear(num_cells, 1)),
    in_keys=["battle"],
)
value_module.to(device)
print("Running policy:", policy_module(env.reset()))
print("Running value:", value_module(env.reset()))

collector = SyncDataCollector(
    make_torchrl_env(env_type="sts-battle"),
    policy_module,
    split_trajs=False,
    device=device,
    frames_per_batch=64,
    total_frames=64000,

)
replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(max_size=10000),
    sampler=SamplerWithoutReplacement(),
)

advantage_module = GAE(
    gamma=gamma, lmbda=lmbda, value_network=value_module, average_gae=True
)

loss_module = A2CLoss(
    actor_network=policy_module,
    critic_network=value_module,
    entropy_bonus=bool(entropy_eps),
    entropy_coef=entropy_eps,
    # these keys match by default but we set this for completeness
    critic_coef=1.0,
    loss_critic_type="smooth_l1",
)

optim = torch.optim.Adam(loss_module.parameters(), lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optim, 1000, 0.0
)

logs = defaultdict(list)
pbar = tqdm(total=64000)
eval_str = ""
learning_starts = 5
# We iterate over the collector until it reaches the total number of frames it was
# designed to collect:
for i, tensordict_data in enumerate(collector):
    # we now have a batch of data to work with. Let's learn something from it.

    advantage_module(tensordict_data)
    data_view = tensordict_data.reshape(-1)
    replay_buffer.extend(data_view.cpu())
    if i > learning_starts:
        for _ in range(num_epochs):
            for _ in range(64 // sub_batch_size):
                subdata = replay_buffer.sample(sub_batch_size)
                loss_vals = loss_module(subdata.to(device))
                loss_value = (
                        loss_vals["loss_objective"]
                        + loss_vals["loss_critic"]
                        + loss_vals["loss_entropy"]
                )

                # Optimization: backward, grad clipping and optimization step
                loss_value.backward()
                # this is not strictly mandatory but it's good practice to keep
                # your gradient norm bounded
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
                optim.step()
                optim.zero_grad()

    logs["reward"].append(tensordict_data["next", "reward"].mean().item())
    pbar.update(tensordict_data.numel())
    cum_reward_str = (
        f"average reward={logs['reward'][-1]: 4.4f} (init={logs['reward'][0]: 4.4f})"
    )
    logs["step_count"].append(tensordict_data["step_count"].max().item())
    stepcount_str = f"step count (max): {logs['step_count'][-1]}"
    logs["lr"].append(optim.param_groups[0]["lr"])
    lr_str = f"lr policy: {logs['lr'][-1]: 4.4f}"
    if i % 10 == 0:
        # We evaluate the policy once every 10 batches of data.
        # Evaluation is rather simple: execute the policy without exploration
        # (take the expected value of the action distribution) for a given
        # number of steps (1000, which is our ``env`` horizon).
        # The ``rollout`` method of the ``env`` can take a policy as argument:
        # it will then execute this policy at each step.
        with set_exploration_type(InteractionType.MODE), torch.no_grad():
            # execute a rollout with the trained policy
            eval_rollout = env.rollout(1000, policy_module)
            logs["eval reward"].append(eval_rollout["next", "reward"].mean().item())
            logs["eval reward (sum)"].append(
                eval_rollout["next", "reward"].sum().item()
            )
            logs["eval step_count"].append(eval_rollout["step_count"].max().item())
            eval_str = (
                f"eval cumulative reward: {logs['eval reward (sum)'][-1]: 4.4f} "
                f"(init: {logs['eval reward (sum)'][0]: 4.4f}), "
                f"eval step-count: {logs['eval step_count'][-1]}"
            )
            del eval_rollout
    pbar.set_description(", ".join([eval_str, cum_reward_str, stepcount_str, lr_str]))

    # We're also using a learning rate scheduler. Like the gradient clipping,
    # this is a nice-to-have but nothing necessary for PPO to work.
    scheduler.step()
