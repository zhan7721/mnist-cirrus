from collections import deque
from abc import ABC, abstractmethod
from pathlib import Path
import os
import yaml
from hashlib import sha256

from datasets import load_dataset
import matplotlib.pyplot as plt
from transformers import get_linear_schedule_with_warmup
import torch
from torchinfo import summary
from torchview import draw_graph
from rich.console import Console
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import humanhash
import wandb

from util.remote import wandb_update_config, wandb_init
from util.plotting import plot_first_batch
from collators import get_collator


class Trainer(ABC):
    """
    Trainer class for training and evaluating models
    """

    def __init__(
        self,
        accelerator,
        config,
        model_class,
        silent=False,
    ):
        self.console = Console()
        self.silent = silent
        self.accelerator = accelerator

        self.console_rule("Accelerator")
        self.console_print(f"[green]device[/green]: {self.accelerator.device}")
        self.console_print(
            f"[green]num_processes[/green]: {self.accelerator.num_processes}"
        )
        self.console_print(
            f"[green]distributed type[/green]: {self.accelerator.distributed_type}"
        )

        training_args = config["training"]
        model_args = config["model"]
        collator_args = config["collator"]

        # hash config
        config_hash = sha256(yaml.dump(config).encode()).hexdigest()
        self.console_rule("Config Hash")
        self.console_print(f"[green]config hash[/green]: {config_hash}")
        hh = humanhash.humanize(config_hash)
        self.console_print(f"[green]human hash[/green]: {hh}")
        if config["training"].run_name is None:
            training_args.run_name = hh
            if not Path(training_args.checkpoint_path).exists():
                Path(training_args.checkpoint_path).mkdir(parents=True, exist_ok=True)
            if (Path(training_args.checkpoint_path) / training_args.run_name).exists():
                # add a number to the run name if it already exists
                num_runs = len(
                    list(
                        Path(training_args.checkpoint_path).glob(
                            f"{training_args.run_name}*"
                        )
                    )
                )
                training_args.run_name = f"{training_args.run_name}-{num_runs}"
            self.console_print(
                f"[green]run name[/green]: {training_args.run_name} (generated)"
            )
        else:
            self.console_print(f"[green]run name[/green]: {training_args.run_name}")

        self.console_rule("Configurations")
        self.console_print("Training Args")
        self.console_print(training_args)
        self.console_print("Model Args")
        self.console_print(model_args)
        self.console_print("Collator Args")
        self.console_print(collator_args)
        self.collator = get_collator(collator_args)
        self.model = model_class(model_args)
        if training_args.optimizer == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=training_args.lr,
            )
        else:
            raise ValueError(f"Optimizer {training_args.optimizer} not implemented")
        if training_args.lr_schedule == "linear":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=training_args.n_steps
            )
        elif training_args.lr_schedule == "linear_with_warmup":
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=training_args.lr_warmup_steps,
                num_training_steps=training_args.n_steps,
            )
        else:
            raise ValueError(f"Scheduler {training_args.lr_schedule} not implemented")
        if training_args.loss == "cross_entropy":
            self.loss_func = torch.nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Loss {training_args.loss} not implemented")
        self.collator_args = collator_args
        self.training_args = training_args
        self.global_step = 0

        self.print_and_draw_model()

        train_ds = load_dataset(training_args.dataset, split=training_args.train_split)
        val_ds = load_dataset(training_args.dataset, split=training_args.val_split)

        self.train_dl = DataLoader(
            train_ds,
            batch_size=training_args.batch_size,
            shuffle=True,
            collate_fn=self.collator,
            drop_last=True,
        )

        self.val_dl = DataLoader(
            val_ds,
            batch_size=training_args.batch_size,
            shuffle=False,
            collate_fn=self.collator,
        )

        self.console_rule("Dataset")
        self.console_print(f"[green]dataset[/green]: {training_args.dataset}")
        self.console_print(f"[green]train_split[/green]: {training_args.train_split}")
        self.console_print(f"[green]val_split[/green]: {training_args.val_split}")
        self.console_print(f"[green]train samples[/green]: {len(train_ds)}")
        self.console_print(f"[green]val sample[/green]: {len(val_ds)}")

        if accelerator.is_main_process:
            first_batch = self.collator(
                [train_ds[i] for i in range(training_args.batch_size)]
            )
            plot_first_batch(first_batch, training_args)
            plt.savefig(
                f"{self.training_args.figure_path}/first_batch.png"
            )  # save the plot

        # wandb
        if accelerator.is_main_process:
            wandb_name, wandb_project, wandb_path, wandb_mode = (
                training_args.run_name,
                training_args.wandb_project,
                training_args.wandb_path,
                training_args.wandb_mode,
            )
            wandb_init(wandb_name, wandb_project, wandb_path, wandb_mode)
            wandb.run.log_code()
            wandb_update_config(
                {
                    "training": training_args,
                    "model": model_args,
                }
            )

        self.model, self.optimizer, self.train_dl, self.val_dl, self.scheduler = (
            accelerator.prepare(
                self.model, self.optimizer, self.train_dl, self.val_dl, self.scheduler
            )
        )

    def print_and_draw_model(self):
        """
        Print and draw the model
        """
        self.console_rule("Model")
        bsz = self.training_args.batch_size
        dummy_input = self.model.dummy_input
        # repeat dummy input to match batch size (regardless of how many dimensions)
        if isinstance(dummy_input, torch.Tensor):
            dummy_input = dummy_input.repeat(
                (bsz,) + (1,) * (len(dummy_input.shape) - 1)
            )
            self.console_print(f"[green]input shape[/green]: {dummy_input.shape}")
        elif isinstance(dummy_input, list):
            dummy_input = [
                x.repeat((bsz,) + (1,) * (len(x.shape) - 1)) for x in dummy_input
            ]
            self.console_print(
                f"[green]input shapes[/green]: {[x.shape for x in dummy_input]}"
            )
        model_summary = summary(
            self.model,
            input_data=dummy_input,
            verbose=0,
            col_names=[
                "input_size",
                "output_size",
                "num_params",
            ],
        )
        self.console_print(model_summary)
        Path(self.training_args.figure_path).mkdir(exist_ok=True)
        if self.accelerator.is_main_process:
            _ = draw_graph(
                self.model,
                input_data=dummy_input,
                save_graph=True,
                directory="figures/",
                filename="model",
                expand_nested=True,
            )
            # remove "figures/model" file
            os.remove(f"{self.training_args.figure_path}/model")

    def wandb_log(self, prefix, log_dict, round_n=3, print_log=True):
        """
        Log to wandb and print to console
        """
        if self.accelerator.is_main_process:
            log_dict = {f"{prefix}/{k}": v for k, v in log_dict.items()}
            wandb.log(log_dict, step=self.global_step)
            if print_log and not self.silent:
                log_dict = {k: round(v, round_n) for k, v in log_dict.items()}
                self.console.log(log_dict)

    def save_checkpoint(self, name_override=None):
        """
        Save model and training args to checkpoint

        Args:
            name_override (str, optional): name to override the default name. Defaults to None.
        """
        self.accelerator.wait_for_everyone()
        checkpoint_name = self.training_args.run_name
        if name_override is not None:
            name = name_override
        else:
            name = f"step_{self.global_step}"
        checkpoint_path = (
            Path(self.training_args.checkpoint_path) / checkpoint_name / name
        )
        if name_override is None:
            # remove old checkpoints
            if checkpoint_path.exists():
                for f in checkpoint_path.iterdir():
                    os.remove(f)
        # model
        if not checkpoint_path.exists():
            checkpoint_path.mkdir(parents=True, exist_ok=True)
        self.model.save_model(checkpoint_path, self.accelerator)
        if self.accelerator.is_main_process:
            # training args
            with open(
                checkpoint_path / "training_args.yml", "w", encoding="utf-8"
            ) as f:
                f.write(yaml.dump(self.training_args.__dict__, Dumper=yaml.Dumper))
            # collator args
            with open(
                checkpoint_path / "collator_args.yml", "w", encoding="utf-8"
            ) as f:
                f.write(yaml.dump(self.collator_args.__dict__, Dumper=yaml.Dumper))
        self.accelerator.wait_for_everyone()
        return checkpoint_path

    def create_latest_model_for_eval(self, device="cpu"):
        """
        Create a model from the latest checkpoint for evaluation

        Args:
            device (str, optional): device to move the model to. Defaults to "cpu".
        """
        checkpoint_path = self.save_checkpoint("latest")
        if self.accelerator.is_main_process:
            eval_model = type(self.model).from_pretrained(checkpoint_path)
            eval_model.eval()
            eval_model = eval_model.to(device)
        return eval_model

    def console_print(self, *args, **kwargs):
        """
        Print to console
        """
        if self.accelerator.is_main_process and not self.silent:
            self.console.print(*args, **kwargs)

    def console_rule(self, *args, **kwargs):
        """
        Print rule to console
        """
        if self.accelerator.is_main_process and not self.silent:
            self.console.rule(*args, **kwargs)

    def evaluate(self):
        """
        Evaluate the model
        """
        self.console_rule("Evaluating")
        self.model.eval()
        device = self.training_args.eval_device
        if device is None:
            device = self.accelerator.device
        if self.accelerator.is_main_process:
            if self.training_args.do_full_eval:
                self.console_print("Full evaluation")
                eval_model = self.create_latest_model_for_eval(device)
                self.evaluate_full(eval_model, device)
            self.console_print("Evaluating loss only")
            # we don't need to move the model to device if we're only evaluating the loss
            self.evaluate_loss_only(self.model, self.accelerator.device)
        self.model.train()

    def train(self):
        """
        Train the model
        """
        if hasattr(self.train_dl, "__len__"):
            n_epochs = self.training_args.n_steps // len(self.train_dl) + 1
            self.console_print(f"[green]expected number of epochs[/green]: {n_epochs}")
        else:
            n_epochs = None
        self.console_print(
            f"[green]effective batch size[/green]: {self.training_args.batch_size*self.accelerator.num_processes}"
        )
        pbar = tqdm(total=self.training_args.n_steps, desc="step")
        losses = deque(maxlen=self.training_args.log_every_n_steps)
        last_losses = None
        stop_loop = False
        while True:
            for batch in self.train_dl:
                global_step = self.global_step
                with self.accelerator.accumulate(self.model):
                    ls = self.train_step(batch, self.accelerator.device)
                losses.append({k: v.detach() for k, v in ls.items()})
                # log
                if (
                    global_step > 0
                    and global_step % self.training_args.log_every_n_steps == 0
                    and self.accelerator.is_main_process
                ):
                    last_losses = {
                        k: torch.mean(torch.tensor([l[k] for l in losses])).item()
                        for k in losses[0].keys()
                    }
                    self.wandb_log("train", last_losses, print_log=False)
                    # log learning rate
                    self.wandb_log(
                        "train",
                        {"lr": self.optimizer.param_groups[0]["lr"]},
                        print_log=False,
                    )
                # save
                if (
                    self.training_args.do_save
                    and global_step > 0
                    and global_step % self.training_args.save_every_n_steps == 0
                ):
                    self.save_checkpoint()
                # stop
                if (
                    self.training_args.n_steps is not None
                    and global_step >= self.training_args.n_steps
                ):
                    stop_loop = True
                    break
                # eval
                if (
                    self.training_args.eval_every_n_steps is not None
                    and global_step > 0
                    and global_step % self.training_args.eval_every_n_steps == 0
                    and self.accelerator.is_main_process
                ):
                    self.evaluate()
                if self.accelerator.is_main_process:
                    self.global_step += 1
                    pbar.update(1)
                    if last_losses is not None:
                        pbar.set_postfix(last_losses)
            if stop_loop:
                break

        # final evaluation
        self.evaluate()

        # save final checkpoint
        if self.training_args.do_save:
            self.save_checkpoint("final")

        if (
            self.accelerator.is_main_process
            and self.training_args.wandb_mode == "offline"
        ):
            self.console_rule("Weights & Biases")
            self.console_print(
                f"use \n[magenta]wandb sync {Path(wandb.run.dir).parent}[/magenta]\nto sync offline run"
            )

    @abstractmethod
    def train_step(self, batch, device):
        """
        Train for one step
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate_full(self, eval_model, device):
        raise NotImplementedError

    @abstractmethod
    def evaluate_loss_only(self, eval_model, device):
        raise NotImplementedError
