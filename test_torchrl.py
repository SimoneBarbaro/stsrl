import json
import logging
import os
from stsrl.trainer.torchrl_actor_critic import TorchPPOExperiment

logger = logging.getLogger(__name__)
dir_path = os.path.dirname(os.path.realpath(__file__))
logging.basicConfig(filename=os.path.join(dir_path, "logs", f"test-torchrl.log"),
                    level=logging.DEBUG,
                    filemode="w",)


def main():
    config = json.load(open("configs/ppo_test.json"))
    config["experiment_dir"] = os.path.join(dir_path, config["experiment_dir"])
    experiment = TorchPPOExperiment(config)

    experiment.init_models()

    experiment.train_battle()

    experiment.train_game()


if __name__ == "__main__":
    main()
