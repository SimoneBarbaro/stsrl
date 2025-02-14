import stsrl.slaythespire as sts


class BattleContextPickler:
    @staticmethod
    def pickle_battle_context(obj):
        return sts.BattleContext, ()

    @staticmethod
    def unpickle_game_context(cls, state):
        return cls()


class GameContextPickler:
    @staticmethod
    def pickle_game_context(obj):
        return sts.GameContext, (obj.character_class, obj.seed, obj.ascension)

    @staticmethod
    def unpickle_game_context(cls, state):
        return cls(*state)


# Register the custom pickler
sts.GameContext.__reduce__ = GameContextPickler.pickle_game_context
sts.BattleContext.__reduce__ = BattleContextPickler.pickle_battle_context
