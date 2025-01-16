import sys
import tqdm
import stsrl.slaythespire as sts
from stsrl.game_encoding import StsEncodings
from skrl.envs.wrappers.torch import wrap_env
import gymnasium
import gymnasium as gym
import json
import numpy as np
from typing import Optional

# import the skrl components to build the RL system
from skrl.agents.torch.a2c import A2C, A2C_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import Trainer
import torch
import torch.nn as nn
from torchrl.modules import MaskedCategorical
from skrl.models.torch import Model, CategoricalMixin, DeterministicMixin


class StsBattleEnvironment(gymnasium.Env):
    def __init__(self):
        super().__init__()
        self.gc = None
        self.bc = None
        self.observation_space = gym.spaces.Box(np.zeros_like(StsEncodings.nniInstance.getBattleObservationMaximums()), np.array(StsEncodings.nniInstance.getBattleObservationMaximums()))
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
        if (action.value in [a.value for a in actions]):
            self.bc.execute(action)
        else:
            raise NotImplementedError(f"Somehow we got a invalid action: {action.print_desc(self.bc)}, expected: {[a.print_desc(self.bc) for a in actions]}")

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

        self.observation_space = gym.spaces.Box(np.zeros_like(StsEncodings.nniInstance.getObservationMaximums()), np.array(StsEncodings.nniInstance.getObservationMaximums()))
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
        if (action.value in [a.value for a in actions]):
            self.gc.execute(action)
        else:
            raise NotImplementedError(f"Somehow we got a invalid action: {action.print_desc(self.gc)}, expected: {[a.print_desc(self.gc) for a in actions]}")

        if self.gc.outcome == sts.GameOutcome.PLAYER_VICTORY:
            reward = 1
        if self.gc.outcome == sts.GameOutcome.PLAYER_LOSS:
            reward = -1
        return self._get_obs(), reward, self.gc.outcome != sts.GameOutcome.UNDECIDED, truncated, self._get_info()

"""
class StsEnvironment(gymnasium.Env):
    def __init__(self):
        self.gc = sts.GameContext(sts.CharacterClass.IRONCLAD, 0, 0)
        self.bc = None

        self.observation_space = gym.spaces.Dict(
            {
                # There is probably a better way to describe the environment
                "game": gym.spaces.Box(np.zeros_like(StsEncodings.nniInstance.getObservationMaximums()), np.array(StsEncodings.nniInstance.getObservationMaximums())),
                "battle": gym.spaces.Box(np.zeros_like(StsEncodings.nniInstance.getBattleObservationMaximums()), np.array(StsEncodings.nniInstance.getBattleObservationMaximums())),
            }
        )
        #self.action_space = gym.spaces.Dict(
        #    {
        #        "game": gym.spaces.Tuple((
        #            gym.spaces.Discrete(13), # action type
        #            gym.spaces.Discrete(7), # reward type
        #            gym.spaces.Discrete(7), # idx 1
        #            gym.spaces.Discrete(5) # idx 2
        #            )),
        #        "battle": gym.spaces.Tuple((
        #            gym.spaces.Discrete(5), # action type
        #            gym.spaces.Discrete(10), # source idx
        #            gym.spaces.Discrete(5), # target idx 
        #            ))
        #    })
        # TODO Should it be better to manually concatenate possible actions?
        self.action_space = gym.spaces.MultiDiscrete([
            13, # action type / source idx
            7, # reward type / action type
            7, # idx 1 / target idx
            5 # idx 2 / nothing
            ])

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
            "game_actions": [StsEncodings.encode_game_action(self.gc, a) for a in self.gc.get_available_actions()],
            "battle_actions": [StsEncodings.encode_battle_action(a) for a in self.bc.get_available_actions()] if self.gc.screen_state == sts.ScreenState.BATTLE else []
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

    def _battle_step(self, action):
        actions = self.bc.get_available_actions()
        action = sts.SearchAction(
            sts.SearchActionType(action[1]),
            action[0], action[2])
        
        if (action.value in [a.v for a in actions]):
            action = actions[action]
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
            actions = self.gc.get_available_actions()
            action = sts.GameAction(
                sts.GameActionType(action[0]),
                sts.RewardActionType(action[1]),
                action[2], action[3])
            if (action.value in [a.v for a in actions]):
                action = actions[action]
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

gym.register(id="sts", entry_point=StsEnvironment)
"""
gym.register(id="sts-game", entry_point=StsGameEnvironment)
gym.register(id="sts-battle", entry_point=StsBattleEnvironment)


class StsGameAgent(A2C):
    """Agent using sts_lightspeed simulator to make choices in the game"""
    def __init__(self, models, memory = None, observation_space = None, action_space = None, device = None, cfg = None):
        super().__init__(models, memory, observation_space, action_space, device, cfg)
        self.base_policy = sts.Agent()
        self.base_policy.print_actions = False
        self.base_policy.print_logs = False
        self.mask = torch.zeros(self.action_space.n).to(self.device)

    def act(self, states, timestep, timesteps, info=None):
        actions, log_prob, outputs = super().act(states, timestep, timesteps)
        # Filter outputs by legal actions
        distribution = MaskedCategorical(
            logits=outputs["net_output"], 
            mask=self.mask.scatter(0, torch.tensor(info['legal_actions']).to(actions.device), 1).bool())
        actions = distribution.sample(actions.shape[:1])
        log_prob = distribution.log_prob(actions)
        return actions, log_prob, outputs


class StsTrainer(Trainer):
    def __init__(self, env: StsGameEnvironment, battle_env: StsBattleEnvironment, agent, battle_agent, agents_scope = None, cfg = None):
        super().__init__(env, agent, agents_scope, cfg)
        self.game_agent = agent
        self.battle_agent = battle_agent
        self.battle_env = battle_env
        self.game_agent.init(cfg)
        self.battle_agent.init(cfg)


    def playout_battle(self, timestep):
        self.battle_env.set_gc(self.env.gc)
        states, infos = self.battle_env.reset()
        terminated = torch.zeros(self.env.num_envs, device=self.battle_env.device).bool()
        while not terminated.any():
            # pre-interaction
            self.battle_agent.pre_interaction(timestep=timestep, timesteps=self.timesteps)

            # compute actions
            with torch.no_grad():
                actions = self.battle_agent.act(states, timestep=timestep, timesteps=self.timesteps, info=infos)[0]

                # step the environments
                next_states, rewards, terminated, truncated, infos = self.battle_env.step(actions)

                # render scene
                if not self.headless:
                    self.battle_env.render()

                # record the environments' transitions
                self.battle_agent.record_transition(states=states,
                                            actions=actions,
                                            rewards=rewards,
                                            next_states=next_states,
                                            terminated=terminated,
                                            truncated=truncated,
                                            infos=infos,
                                            timestep=timestep,
                                            timesteps=self.timesteps)

                # log environment info
                if self.environment_info in infos:
                    for k, v in infos[self.environment_info].items():
                        if isinstance(v, torch.Tensor) and v.numel() == 1:
                            self.game_agent.track_data(f"Info / {k}", v.item())

            # post-interaction
            self.battle_agent.post_interaction(timestep=timestep, timesteps=self.timesteps)
        self.env.step_out_of_combat(infos["bc"])


    def train(self):
        # set running mode
        self.game_agent.set_running_mode("train")
        self.battle_agent.set_running_mode("train")

        # reset env
        states, infos = self.env.reset()

        for timestep in tqdm.tqdm(range(self.initial_timestep, self.timesteps), disable=self.disable_progressbar, file=sys.stdout):
            if self.env.is_battle():
                self.playout_battle(timestep)
            else:
                # pre-interaction
                self.game_agent.pre_interaction(timestep=timestep, timesteps=self.timesteps)

                # compute actions
                with torch.no_grad():
                    actions = self.game_agent.act(states, timestep=timestep, timesteps=self.timesteps, info=infos)[0]

                    # step the environments
                    next_states, rewards, terminated, truncated, infos = self.env.step(actions)

                    # render scene
                    if not self.headless:
                        self.env.render()

                    # record the environments' transitions
                    self.game_agent.record_transition(states=states,
                                                actions=actions,
                                                rewards=rewards,
                                                next_states=next_states,
                                                terminated=terminated,
                                                truncated=truncated,
                                                infos=infos,
                                                timestep=timestep,
                                                timesteps=self.timesteps)

                    # log environment info
                    if self.environment_info in infos:
                        for k, v in infos[self.environment_info].items():
                            if isinstance(v, torch.Tensor) and v.numel() == 1:
                                self.game_agent.track_data(f"Info / {k}", v.item())

                # post-interaction
                self.game_agent.post_interaction(timestep=timestep, timesteps=self.timesteps)

            # reset environments
            if self.env.num_envs > 1:
                states = next_states
            else:
                if terminated.any() or truncated.any():
                    with torch.no_grad():
                        states, infos = self.env.reset()
                else:
                    states = next_states

    def eval(self):
        return super().eval()


class ActorMLP(CategoricalMixin, Model):
    def __init__(self, observation_space, action_space, device, unnormalized_log_prob=True):
        Model.__init__(self, observation_space, action_space, device)
        CategoricalMixin.__init__(self, unnormalized_log_prob)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 32),
                                 nn.ReLU(),
                                 nn.Linear(32, self.num_actions))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}


class CriticMLP(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations# + self.num_actions
                                           , 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 32),
                                 nn.ReLU(),
                                 nn.Linear(32, 1))

    def compute(self, inputs, role):
        return self.net(torch.cat([inputs["states"]#, inputs["taken_actions"]
                                   ], dim=1)), {}
env = gym.make('sts-game')
# wrap the environment
env = wrap_env(env)
battle_env = gym.make('sts-battle')
# wrap the environment
battle_env = wrap_env(battle_env)

#env._device = "cpu"
#battle_env._device = "cpu"

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
    cfg["experiment"]["directory"] = "runs/test/" + agent_name

    # instantiate the agent
    # (assuming a defined environment <env> and memory <memory>)

    class MyMemory(RandomMemory):
        def create_tensor(self, name, size, dtype = None, keep_dimensions = False):
            return super().create_tensor(name, size, dtype, keep_dimensions)

    # instantiate a memory as experience replay
    memory = MyMemory(memory_size=cfg["learning_starts"], num_envs=env.num_envs, device=env.device, replacement=False)

    # initialize models' parameters (weights and biases)
    for model in models.values():
        model.init_parameters(method_name="normal_", mean=0.0, std=0.1)


    return StsGameAgent(models=models,
                memory=memory,
                cfg=cfg,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=env.device)



# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 50000, "headless": True}
#trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=[agent])
trainer = StsTrainer(
    cfg=cfg_trainer, env=env, battle_env=battle_env, agent=get_agent(env, "game_a2c"), battle_agent=get_agent(battle_env, "battle_a2c"))

# start training
trainer.train()
