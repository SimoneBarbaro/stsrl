import torch
from skrl.agents.torch.a2c import A2C
from torchrl.modules import MaskedCategorical
import stsrl.slaythespire as sts


class StsGameAgent(A2C):
    """Agent using sts_lightspeed simulator to make choices in the game"""

    def __init__(self, models, memory=None, observation_space=None, action_space=None, device=None, cfg=None):
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
