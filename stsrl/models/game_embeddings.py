import torch
import torch.nn
from ray.rllib.core.models.base import Encoder, ENCODER_OUT
from ray.rllib.core.models.configs import ModelConfig
from ray.rllib.core.models.torch.base import TorchModel


class BattleEmbeddingModule(torch.nn.Module):
    def __init__(self, player_embedding_shape=4,
                 player_info_embedding_shape=2,
                 card_embedding_shape=32,
                 status_embedding_shape=16,
                 monster_embedding_shape=16,
                 relics_embedding_shape=16,
                 potions_embedding_shape=8,
                 ) -> None:
        super().__init__()
        self.player_embedding = torch.nn.Linear(9, player_embedding_shape, bias=False)
        self.player_info_embedding = torch.nn.Linear(8, player_info_embedding_shape, bias=False)
        self.status_embedding = torch.nn.Linear(86, status_embedding_shape, bias=False)
        self.card_embedding = torch.nn.Linear(370*2, card_embedding_shape, bias=False)
        self.relics_embedding = torch.nn.Linear(180, relics_embedding_shape, bias=False)
        self.potions_embedding = torch.nn.Linear(43, potions_embedding_shape, bias=False)
        self.monster_embedding = torch.nn.Linear(101, monster_embedding_shape, bias=False)
        self.embedding_models = torch.nn.ModuleList([
            self.player_embedding,
            self.status_embedding,
            self.player_info_embedding,
            self.card_embedding,
            self.card_embedding,
            self.card_embedding,
            self.card_embedding,
            self.card_embedding,
            self.card_embedding,
            self.card_embedding,
            self.card_embedding,
            self.card_embedding,
            self.card_embedding,
            self.card_embedding,
            self.card_embedding,
            self.card_embedding,
            self.potions_embedding,
            self.relics_embedding,
            self.monster_embedding,
            self.monster_embedding,
            self.monster_embedding,
            self.monster_embedding,
            self.monster_embedding,
        ])

        self.embedding_size = player_embedding_shape + player_info_embedding_shape + status_embedding_shape + card_embedding_shape * 13 + potions_embedding_shape + relics_embedding_shape + 5 * monster_embedding_shape

    def forward(self, battle: torch.Tensor):
        inputs = battle.split_with_sizes(
            [9, 86, 8, 370*2, 370*2, 370*2, 370*2, 370*2, 370*2, 370*2, 370*2, 370*2, 370*2, 370*2, 370*2, 370*2, 43, 180, 101, 101, 101, 101, 101], dim=-1)
        return torch.cat([self.embedding_models[i](inputs[i]) for i in range(len(inputs))], dim=-1)


class GameEmbeddingModule(torch.nn.Module):
    def __init__(self, player_embedding_shape=4,
                 boss_embedding_shape=2,
                 card_embedding_shape=32,
                 map_embedding_shape=32,
                 screen_embedding_shape=4,
                 event_embedding_shape=16,
                 relics_embedding_shape=16,
                 potions_embedding_shape=8,
                 prices_embedding_shape=4,
                 ) -> None:
        super().__init__()
        self.player_embedding = torch.nn.Linear(4, player_embedding_shape, bias=False)
        self.boss_embedding = torch.nn.Linear(10, boss_embedding_shape, bias=False)
        self.card_embedding = torch.nn.Linear(370*2, card_embedding_shape, bias=False)
        self.map_embedding = torch.nn.Linear(805, map_embedding_shape, bias=False)
        self.screen_embedding = torch.nn.Linear(9, screen_embedding_shape, bias=False)
        self.event_embedding = torch.nn.Linear(56, event_embedding_shape, bias=False)
        self.relics_embedding = torch.nn.Linear(180, relics_embedding_shape, bias=False)
        self.potions_embedding = torch.nn.Linear(43, potions_embedding_shape, bias=False)
        self.prices_embedding = torch.nn.Linear(15, prices_embedding_shape, bias=False)
        self.embedding_models = torch.nn.ModuleList([
            self.player_embedding,
            self.boss_embedding,
            self.card_embedding,
            self.potions_embedding,
            self.relics_embedding,
            self.map_embedding,
            self.screen_embedding,
            self.event_embedding,
            self.relics_embedding,
            self.card_embedding,
            self.card_embedding,
            self.card_embedding,
            self.card_embedding,
            self.card_embedding,
            self.potions_embedding,
            self.prices_embedding,
        ])
        self.embedding_size = player_embedding_shape + boss_embedding_shape + card_embedding_shape * 6 + 2*potions_embedding_shape + 2*relics_embedding_shape + map_embedding_shape + prices_embedding_shape + event_embedding_shape + screen_embedding_shape

    def forward(self, game: torch.Tensor):
        inputs = game.split_with_sizes(
            [4, 10, 370 * 2, 43, 180, 805, 9, 56, 180, 370 * 2, 370 * 2, 370 * 2, 370 * 2, 370 * 2, 43, 15], dim=-1)
        return torch.cat([self.embedding_models[i](inputs[i]) for i in range(len(inputs))], dim=-1)


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