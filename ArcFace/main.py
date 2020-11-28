import os
from trainer import Trainer
from config.config import Config


os.environ["CUDA_VISIBLE_DEVICES"] = "3"
CUDA: 0


if __name__ == '__main__':
    trainer = Trainer(Config)
    trainer.run()

