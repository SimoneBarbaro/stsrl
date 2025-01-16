import numpy as np
import stsrl.slaythespire as sts

class StsEncodings:
    nniInstance = sts.getNNInterface()
    encodingInstance = sts.getEncodingsInterface()

    def encode_game(gc: sts.GameContext):
        return np.array(StsEncodings.nniInstance.getObservation(gc)).astype(np.float32)
    def encode_battle(gc: sts.GameContext, bc: sts.BattleContext):
        return np.array(StsEncodings.nniInstance.encode_battle(gc, bc)).astype(np.float32)
    def encode_game_action(action: sts.GameAction):
        return StsEncodings.encodingInstance.encode_game_action(action)
    def encode_battle_action(action: sts.SearchAction):
        return StsEncodings.encodingInstance.encode_battle_action(action)
    def decode_game_action(encoding):
        return StsEncodings.encodingInstance.decode_game_action(encoding)
    def decode_battle_action(encoding):
        return StsEncodings.encodingInstance.decode_battle_action(encoding)
