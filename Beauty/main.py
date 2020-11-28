import os
from trainer import Trainer
from tester import Tester
from config.config import Config

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
CUDA: 0

if __name__ == '__main__':
    # trainer = Trainer(Config)
    # trainer.run()

    tester = Tester(Config)
    tester.run()
