import torch


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
        self.monster_embedding = torch.nn.Linear(34, monster_embedding_shape, bias=False)
        self.embedding_size = player_embedding_shape + player_info_embedding_shape + status_embedding_shape + card_embedding_shape * 13 + potions_embedding_shape + relics_embedding_shape + 5 * monster_embedding_shape

    def forward(self, battle: torch.Tensor):
        inputs = battle.split_with_sizes(
            [9, 86, 8, 370*2, 370*2, 370*2, 370*2, 370*2, 370*2, 370*2, 370*2, 370*2, 370*2, 370*2, 370*2, 370*2, 43, 180, 34, 34, 34, 34, 34], dim=-1)
        return torch.cat(
            [
                # TODO constants from sts lib
                self.player_embedding.forward(inputs[0]),
                self.status_embedding.forward(inputs[1]),
                self.player_info_embedding.forward(inputs[2]),
                self.card_embedding.forward(inputs[3]),
                self.card_embedding.forward(inputs[4]),
                self.card_embedding.forward(inputs[5]),
                self.card_embedding.forward(inputs[6]),
                self.card_embedding.forward(inputs[7]),
                self.card_embedding.forward(inputs[8]),
                self.card_embedding.forward(inputs[9]),
                self.card_embedding.forward(inputs[10]),
                self.card_embedding.forward(inputs[11]),
                self.card_embedding.forward(inputs[12]),
                self.card_embedding.forward(inputs[13]),
                self.card_embedding.forward(inputs[14]),
                self.card_embedding.forward(inputs[15]),
                self.potions_embedding.forward(inputs[16]),
                self.relics_embedding.forward(inputs[17]),
                self.monster_embedding.forward(inputs[18]),
                self.monster_embedding.forward(inputs[19]),
                self.monster_embedding.forward(inputs[20]),
                self.monster_embedding.forward(inputs[21]),
                self.monster_embedding.forward(inputs[22]),

            ], dim=-1
        )


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
        self.embedding_size = player_embedding_shape + boss_embedding_shape + card_embedding_shape * 6 + 2*potions_embedding_shape + 2*relics_embedding_shape + map_embedding_shape + prices_embedding_shape + event_embedding_shape + screen_embedding_shape

    def forward(self, game: torch.Tensor):
        inputs = game.split_with_sizes(
            [4, 10, 370 * 2, 43, 180, 805, 9, 56, 180, 370 * 2, 370 * 2, 370 * 2, 370 * 2, 370 * 2, 43, 15], dim=-1)
        return torch.cat(
            [
                # TODO constants from sts lib
                self.player_embedding.forward(inputs[0]),
                self.boss_embedding.forward(inputs[1]),
                self.card_embedding.forward(inputs[2]),
                self.potions_embedding.forward(inputs[3]),
                self.relics_embedding.forward(inputs[4]),
                self.map_embedding.forward(inputs[5]),
                self.screen_embedding.forward(inputs[6]),
                self.event_embedding.forward(inputs[7]),
                self.relics_embedding.forward(inputs[8]),
                self.card_embedding.forward(inputs[9]),
                self.card_embedding.forward(inputs[10]),
                self.card_embedding.forward(inputs[11]),
                self.card_embedding.forward(inputs[12]),
                self.card_embedding.forward(inputs[13]),
                self.potions_embedding.forward(inputs[14]),
                self.prices_embedding.forward(inputs[15]),
            ], dim=-1
        )
