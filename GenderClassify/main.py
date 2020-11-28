from trainer import Trainer
import os

def train():
    trainer = Trainer(
        img_path="./data/IQA_datasets/",
        batch_size=64,
        cuda=True)
    trainer.train(epochs=100)

if __name__ == '__main__':
    train()
