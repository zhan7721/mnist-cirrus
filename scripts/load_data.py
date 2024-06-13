from accelerate import Accelerator
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
import hydra
from omegaconf import DictConfig
import sys
from datasets import load_dataset


sys.path.append(".")  # noqa: E402

from model.simple_mlp import SimpleMLP
from util.simple_trainer import Trainer
from util.util import move_batch_to_device, seed_everything, get_args


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """
    Main function"""
    training_args = get_args(cfg)["training"]

    train_ds = load_dataset(training_args.dataset, split=training_args.train_split)
    val_ds = load_dataset(training_args.dataset, split=training_args.val_split)

# main
if __name__ == "__main__":
    main()