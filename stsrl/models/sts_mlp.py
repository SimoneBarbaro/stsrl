import torch
from torch import nn

from stsrl.models.game_embeddings import BattleEmbeddingModule, GameEmbeddingModule


class SimpleMlpModule(nn.Module):
    def __init__(self, num_cells, num_layers=4, output_len=1, softmax_head=False):
        super().__init__()
        battle_embedding = BattleEmbeddingModule()
        game_embedding = GameEmbeddingModule()
        self.battle_model = nn.Sequential(
            battle_embedding,
            nn.Linear(battle_embedding.embedding_size, num_cells),
            nn.Tanh()
        )
        for _ in range(num_layers - 1):
            self.battle_model.append(nn.Linear(num_cells, num_cells))
            self.battle_model.append(nn.Tanh())
        self.battle_model.append(nn.Linear(num_cells, output_len))
        self.game_model = nn.Sequential(
            game_embedding,
            nn.Linear(game_embedding.embedding_size, num_cells),
            nn.Tanh(),
        )
        for _ in range(num_layers - 1):
            self.game_model.append(nn.Linear(num_cells, num_cells))
            self.game_model.append(nn.Tanh())
        self.game_model.append(nn.Linear(num_cells, output_len))
        if softmax_head:
            self.game_model.append(nn.Softmax())
            self.battle_model.append(nn.Softmax())

    def forward(self, game: torch.Tensor, battle: torch.Tensor, is_battle: torch.Tensor):
        result = is_battle * self.battle_model(battle) + (1 - is_battle) * self.game_model.forward(game)
        return result
