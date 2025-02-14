import time
import sys

import torch
import tqdm
from skrl.trainers.torch import Trainer

from stsrl.environments.game_env import StsGameEnvironment
from stsrl.environments.battle_env import StsBattleEnvironment
import logging

logger = logging.getLogger(__name__)


class StsTrainer(Trainer):
    def __init__(self, env: StsGameEnvironment, battle_env: StsBattleEnvironment, agent, battle_agent,
                 agents_scope=None, cfg=None):
        super().__init__(env, agent, agents_scope, cfg)
        self.game_agent = agent
        self.battle_agent = battle_agent
        self.battle_env = battle_env
        self.game_agent.init(cfg)
        self.battle_agent.init(cfg)
        self.battle_timestep = 0

    def playout_battle(self):
        t_start = time.time()
        self.battle_env.set_gc(self.env.gc)
        states, infos = self.battle_env.reset(options={"from_gc": True})
        t_reset = time.time() - t_start
        self.battle_agent.track_data(f"Timing / t_reset", t_reset)
        terminated = torch.zeros(self.env.num_envs, device=self.battle_env.device).bool()
        while not terminated.any():
            # pre-interaction
            t_start = time.time()
            self.battle_agent.pre_interaction(timestep=self.battle_timestep, timesteps=self.timesteps)
            t_pre_interaction = time.time() - t_start
            # compute actions
            with torch.no_grad():
                t_start = time.time()
                actions = \
                    self.battle_agent.act(
                        states,
                        timestep=self.battle_timestep,
                        timesteps=self.timesteps,
                        info=infos)[0]
                t_act = time.time() - t_start
                # step the environments
                t_start = time.time()
                next_states, rewards, terminated, truncated, infos = self.battle_env.step(actions)
                t_step = time.time() - t_start
                # render scene
                if not self.headless:
                    self.battle_env.render()

                # record the environments' transitions
                t_start = time.time()
                self.battle_agent.record_transition(states=states,
                                                    actions=actions,
                                                    rewards=rewards,
                                                    next_states=next_states,
                                                    terminated=terminated,
                                                    truncated=truncated,
                                                    infos=infos,
                                                    timestep=self.battle_timestep,
                                                    timesteps=self.timesteps)
                t_record_transition = time.time() - t_start

                # log environment info
                if self.environment_info in infos:
                    for k, v in infos[self.environment_info].items():
                        if isinstance(v, torch.Tensor) and v.numel() == 1:
                            self.battle_agent.track_data(f"Info / {k}", v.item())
                        else:
                            self.battle_agent.track_data(f"Info / {k}", v)

            # post-interaction
            t_start = time.time()
            self.battle_agent.post_interaction(timestep=self.battle_timestep, timesteps=self.timesteps)
            t_post_interaction = time.time() - t_start
            self.battle_timestep += 1
            self.battle_agent.track_data(f"Timing / t_pre_interaction", t_pre_interaction)
            self.battle_agent.track_data(f"Timing / t_act", t_act)
            self.battle_agent.track_data(f"Timing / t_step", t_step)
            self.battle_agent.track_data(f"Timing / t_record_transition", t_record_transition)
            self.battle_agent.track_data(f"Timing / t_post_interaction", t_post_interaction)
        return self.env.step_out_of_combat(self.battle_env.bc)

    def train(self):
        # set running mode
        self.game_agent.set_running_mode("train")
        self.battle_agent.set_running_mode("train")

        # reset env
        t_start = time.time()
        states, infos = self.env.reset()
        t_reset = time.time() - t_start
        self.game_agent.track_data(f"Timing / t_pre_interaction", t_reset)
        for timestep in tqdm.tqdm(range(self.initial_timestep, self.timesteps), disable=self.disable_progressbar,
                                  file=sys.stdout):
            if self.env.is_battle():
                next_states, rewards, terminated, truncated, infos = self.playout_battle()
                next_states = torch.tensor(next_states).unsqueeze(0).to(states.device)
                rewards = torch.tensor(rewards).unsqueeze(0).to(states.device)
                terminated = torch.tensor(terminated).unsqueeze(0).to(states.device)
                truncated = torch.tensor(truncated).unsqueeze(0).to(states.device)
            else:
                # pre-interaction
                t_start = time.time()
                self.game_agent.pre_interaction(timestep=timestep, timesteps=self.timesteps)
                t_pre_interaction = time.time() - t_start

                # compute actions
                with torch.no_grad():
                    t_start = time.time()
                    actions = self.game_agent.act(states, timestep=timestep, timesteps=self.timesteps, info=infos)[0]
                    t_act = time.time() - t_start

                    # step the environments
                    t_start = time.time()
                    next_states, rewards, terminated, truncated, infos = self.env.step(actions)
                    t_step = time.time() - t_start

                    # render scene
                    if not self.headless:
                        self.env.render()

                    # record the environments' transitions
                    t_start = time.time()
                    self.game_agent.record_transition(states=states,
                                                      actions=actions,
                                                      rewards=rewards,
                                                      next_states=next_states,
                                                      terminated=terminated,
                                                      truncated=truncated,
                                                      infos=infos,
                                                      timestep=timestep,
                                                      timesteps=self.timesteps)
                    t_record_transition = time.time() - t_start

                    # log environment info
                    if self.environment_info in infos:
                        for k, v in infos[self.environment_info].items():
                            if isinstance(v, torch.Tensor) and v.numel() == 1:
                                self.game_agent.track_data(f"Info / {k}", v.item())
                            else:
                                self.game_agent.track_data(f"Info / {k}", v)

                # post-interaction
                t_start = time.time()
                self.game_agent.post_interaction(timestep=timestep, timesteps=self.timesteps)
                t_post_interaction = time.time() - t_start

                self.game_agent.track_data(f"Timing / t_pre_interaction", t_pre_interaction)
                self.game_agent.track_data(f"Timing / t_act", t_act)
                self.game_agent.track_data(f"Timing / t_step", t_step)
                self.game_agent.track_data(f"Timing / t_record_transition", t_record_transition)
                self.game_agent.track_data(f"Timing / t_post_interaction", t_post_interaction)

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
