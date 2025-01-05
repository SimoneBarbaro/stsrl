import numpy as np
import stsrl.slaythespire as sts

class StsEncodings:
    instance = sts.getNNInterface()

    def encode_game(gc: sts.GameContext):
        return np.array(StsEncodings.instance.observation(gc).astype(np.float32))
    def encode_battle(gc: sts.GameContext, bc: sts.BattleContext):
        return np.array(StsEncodings.instance.encode_battle(gc, bc)).astype(np.float32)
