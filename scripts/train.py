from accelerate import Accelerator
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
import hydra
from omegaconf import DictConfig
import sys

sys.path.append(".")  # noqa: E402

from model.simple_mlp import SimpleMLP
from util.simple_trainer import Trainer
from util.util import move_batch_to_device, seed_everything, get_args


class MyTrainer(Trainer):
    """
    MyTrainer class
    """

    def train_step(self, batch, device):
        inputs = batch["image"]
        labels = batch["target"]
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = self.model(inputs)
        loss = self.loss_func(outputs, labels)
        self.accelerator.backward(loss)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        return {"loss": loss}

    def evaluate_full(self, eval_model, device):
        y_true = []
        y_pred = []
        for batch in self.val_dl:
            batch = move_batch_to_device(batch, device)
            y = eval_model(batch["image"])
            y_true.append(batch["target"].cpu().numpy())
            y_pred.append(y.argmax(-1).cpu().numpy())
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro")
        precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
        recall = recall_score(y_true, y_pred, average="macro")
        self.wandb_log(
            "val", {"acc": acc, "f1": f1, "precision": precision, "recall": recall}
        )

    def evaluate_loss_only(self, eval_model, device):
        losses = []
        for batch in self.val_dl:
            batch = move_batch_to_device(batch, device)
            y = eval_model(batch["image"])
            loss = self.loss_func(y, batch["target"])
            losses.append(loss.detach())
        self.wandb_log("val", {"loss": torch.mean(torch.tensor(losses)).item()})


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """
    Main function"""
    accelerator = Accelerator()

    trainer = MyTrainer(
        accelerator=accelerator,
        config=get_args(cfg),
        model_class=SimpleMLP,
    )

    # evaluation
    if trainer.training_args.eval_only:
        seed_everything(trainer.training_args.seed)
        trainer.evaluate()
    else:
        seed_everything(trainer.training_args.seed)
        trainer.train()


# main
if __name__ == "__main__":
    main()
