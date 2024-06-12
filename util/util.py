import torch
import numpy as np

from configs.args import TrainingArgs, ModelArgs, CollatorArgs


def seed_everything(seed):
    """
    Seed everything for reproducibility
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def move_batch_to_device(batch, device):
    """
    Move batch to device
    """
    if isinstance(batch, dict):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = move_batch_to_device(batch[k], device)
            elif isinstance(v, list):
                # recursively move list of tensors to device
                batch[k] = move_batch_to_device(batch[k], device)
    elif isinstance(batch, list):
        batch = [move_batch_to_device(x, device) for x in batch]
    elif isinstance(batch, torch.Tensor):
        batch = batch.to(device)
    return batch


def get_args(cfg):
    if hasattr(cfg, "training"):
        training_args = TrainingArgs(**cfg.training)
    else:
        training_args = TrainingArgs()
    if hasattr(cfg, "model"):
        model_args = ModelArgs(**cfg.model)
    else:
        model_args = ModelArgs()
    if hasattr(cfg, "collator"):
        collator_args = CollatorArgs(**cfg.collator)
    else:
        collator_args = CollatorArgs()
    return {
        "training": training_args,
        "model": model_args,
        "collator": collator_args,
    }
