import gymnasium as gym

from stsrl.environments.sts_env import StsEnvironment, StsSkipbattleEnvironment
from stsrl.environments.game_env import StsGameEnvironment
from stsrl.environments.battle_env import StsBattleEnvironment

gym.register(id="sts", entry_point=StsEnvironment)
gym.register(id="sts-game", entry_point=StsGameEnvironment)
gym.register(id="sts-battle", entry_point=StsBattleEnvironment)
gym.register(id="sts-skip-battle", entry_point=StsSkipbattleEnvironment)
