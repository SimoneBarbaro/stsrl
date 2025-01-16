import sys

import torch
import tqdm
from skrl.trainers.torch import Trainer

from stsrl.gym.environments import StsGameEnvironment, StsBattleEnvironment


class StsTrainer(Trainer):
    def __init__(self, env: StsGameEnvironment, battle_env: StsBattleEnvironment, agent, battle_agent,
                 agents_scope=None, cfg=None):
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

        for timestep in tqdm.tqdm(range(self.initial_timestep, self.timesteps), disable=self.disable_progressbar,
                                  file=sys.stdout):
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