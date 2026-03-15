from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import wandb
import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, gather_object
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

from urgent_mos.utils import calculate_metrics

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        exp_dir: Path | str,
        pretrained_ckpt: Path = None,
        epochs: int = 100,
        learning_rate: float = 5e-5,
        num_warmup_updates: int = 2500,
        grad_accumulation_steps: int = 1,
        grad_norm: float = 1.0,
        save_per_updates: int = 1000,
        log_per_updates: int = 100,
        keep_last_n_checkpoints: int = -1,
        accelerate_kwargs: dict = dict(),
        config_str: Optional[str] = None,
        report_to: Optional[str] = None,
        scheduler_type: str = "linear",
    ):
        # NOTE: a model_config_str is taken here to prevent Hydra from instantiating the model config
        self.pretrained_ckpt = pretrained_ckpt
        self.exp_dir = Path(exp_dir)
        self.config = json.loads(config_str)

        self.model = model

        self.epochs = epochs
        self.num_warmup_updates = num_warmup_updates
        self.save_per_updates = save_per_updates
        self.keep_last_n_checkpoints = keep_last_n_checkpoints
        self.log_per_updates = log_per_updates

        self.learning_rate = learning_rate
        self.grad_accumulation_steps = grad_accumulation_steps
        self.grad_norm = grad_norm
        self.scheduler_type = scheduler_type

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(
            log_with=report_to,
            kwargs_handlers=[ddp_kwargs],
            gradient_accumulation_steps=self.grad_accumulation_steps,
            **accelerate_kwargs,
        )

        model_cfg_dict = {
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "num_warmup_updates": self.num_warmup_updates,
            "grad_accumulation_steps": self.grad_accumulation_steps,
            "grad_norm": self.grad_norm,
            "gpus": self.accelerator.num_processes,
        }
        self.accelerator.init_trackers(
            project_name=os.getenv("WANDB_PROJECT", "urgentmos"),
            config=model_cfg_dict,
            init_kwargs={"wandb": {"name": self.exp_dir.stem}},  # Set custom run name here
        )

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def save_checkpoint(self, update, is_last=False):
        if not self.is_main:
            return

        checkpoint = dict(
            model=self.accelerator.unwrap_model(self.model).state_dict(),
            optimizer=self.optimizer.state_dict(),
            scheduler=self.scheduler.state_dict(),
            update=update,
            config=self.config,
        )
        if is_last:
            self.accelerator.save(checkpoint, self.exp_dir / "model_last.pt")
            logger.info(f"Saved last checkpoint at update {update}")
            return

        if self.keep_last_n_checkpoints == 0:
            return

        self.accelerator.save(checkpoint, f"{self.exp_dir}/model_{update}.pt")
        if self.keep_last_n_checkpoints > 0:
            checkpoints = [ckpt for ckpt in self.exp_dir.glob("*.pt") if ckpt.stem != "model_last"]
            checkpoints.sort(key=lambda p: int(p.stem.removeprefix("model_")))
            while len(checkpoints) > self.keep_last_n_checkpoints:
                earliest_checkpoint = checkpoints.pop(0)
                earliest_checkpoint.unlink()
                logger.info(f"Removed early checkpoint: {earliest_checkpoint}")

    def load_checkpoint(self):
        self.accelerator.wait_for_everyone()

        latest_checkpoint = None
        if self.pretrained_ckpt is not None:
            latest_checkpoint = self.pretrained_ckpt

        if (self.exp_dir / "model_last.pt").exists():
            latest_checkpoint = self.exp_dir / "model_last.pt"
        else:
            ckpts = list(self.exp_dir.glob("*.pt"))
            if len(ckpts) > 0:
                latest_checkpoint = max(ckpts, key=lambda p: int(p.stem.removeprefix("model_")), default=None)

        if latest_checkpoint is None:
            logger.info("No checkpoint found, starting from scratch.")
            return 0

        checkpoint = torch.load(latest_checkpoint, weights_only=True, map_location="cpu")

        if latest_checkpoint == self.pretrained_ckpt:
            self.model.load_state_dict(checkpoint["model"])
            update = 0
        else:
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.scheduler.load_state_dict(checkpoint["scheduler"])
            update = checkpoint["update"]

        return update

    @staticmethod
    def get_checkpoint_config(checkpoint_path: Union[Path, str]) -> Optional[Dict[str, Any]]:
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
        return checkpoint.get("model_config", None)

    def train(self, train_dataloader: DataLoader, cv_dataloader: DataLoader, seed: int = 42):
        generator = torch.Generator()
        generator.manual_seed(seed)

        self.optimizer = AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.learning_rate,
            weight_decay=0.01,
        )
        total_updates = math.ceil(len(train_dataloader) / self.grad_accumulation_steps) * self.epochs
        if self.scheduler_type == "cosine":
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.num_warmup_updates,
                num_training_steps=total_updates,
            )
        elif self.scheduler_type == "linear":
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.num_warmup_updates,
                num_training_steps=total_updates,
            )
        else:
            raise ValueError(f"Invalid scheduler type: {self.scheduler_type}")

        current_update = self.load_checkpoint()

        self.accelerator.even_batches = False
        train_dataloader, cv_dataloader = self.accelerator.prepare(train_dataloader, cv_dataloader)
        self.model, self.optimizer, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.scheduler
        )

        orig_epoch_step = len(train_dataloader)
        current_step = current_update * self.grad_accumulation_steps
        skipped_epoch = int(current_step // orig_epoch_step)

        for epoch in range(skipped_epoch, self.epochs):
            self.model.train()
            if epoch == skipped_epoch:
                skipped_batch = current_step % orig_epoch_step
                progress_bar_initial = math.ceil(skipped_batch / self.grad_accumulation_steps)
                current_dataloader = self.accelerator.skip_first_batches(train_dataloader, num_batches=skipped_batch)
            else:
                progress_bar_initial = 0
                current_dataloader = train_dataloader

            # Set epoch for the batch sampler if it exists
            if hasattr(train_dataloader, "batch_sampler") and hasattr(train_dataloader.batch_sampler, "set_epoch"):
                train_dataloader.batch_sampler.set_epoch(epoch)

            progress_bar = tqdm(
                range(math.ceil(len(train_dataloader) / self.grad_accumulation_steps)),
                desc=f"Epoch {epoch + 1}/{self.epochs}",
                unit="update",
                disable=not self.is_main,
                initial=progress_bar_initial,
            )

            for batch in current_dataloader:
                with self.accelerator.accumulate(self.model):
                    loss, info, others = self.model(**batch)
                    self.accelerator.backward(loss)

                    if self.grad_norm > 0 and self.accelerator.sync_gradients:
                        grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                            logger.warning("Gradient norm is NaN or INF. Skipping update.")
                            self.optimizer.zero_grad()
                            continue

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                # Increment current_update on all processes to keep them synchronized
                if self.accelerator.sync_gradients:
                    current_update += 1

                if self.is_main:
                    if self.accelerator.sync_gradients:
                        progress_bar.set_postfix(update=str(current_update), loss=loss.item())
                        progress_bar.update(1)
                        if current_update % self.log_per_updates == 0:
                            text = "\n".join(f"  {k}: {v:.4f}" for k, v in info.items())
                            tqdm.write(f"[step {current_update}]\n{text}")
                    self.accelerator.log(
                        {"loss": loss.item(), "lr": self.scheduler.get_last_lr()[0], **info},
                        step=current_update,
                    )

                if current_update % self.save_per_updates == 0 and self.accelerator.sync_gradients:
                    cv_loss, cv_info = self.cv(cv_dataloader)
                    self.model.train()

                    if self.is_main:
                        self.accelerator.log(
                            {f"cv_{key}": value for key, value in cv_info.items()}, step=current_update
                        )
                        progress_bar.set_postfix(update=str(current_update), loss=cv_loss.item())
                        progress_bar.update(0)
                        text = "\n".join(f"  {k}: {v:.4f}" for k, v in cv_info.items())
                        tqdm.write(f"[step {current_update} (CV)]\n{text}")
                        self.save_checkpoint(current_update)

        self.save_checkpoint(current_update, is_last=True)
        self.accelerator.end_training()

    @torch.inference_mode()
    def cv(self, cv_dataloader: DataLoader):
        self.model.eval()
        absolute_score_metric2preds, absolute_score_metric2refs = {}, {}
        comparative_score_metric2info = {}

        loss = 0.0
        for batch in tqdm(cv_dataloader, desc="Validation", disable=not self.accelerator.is_local_main_process):
            loss_, info, others = self.model(**batch)
            loss += loss_
            if "absolute_score" in others:
                batch_absolute_score_metric2preds = others["absolute_score"]["metric2preds"]
                for name in batch_absolute_score_metric2preds.keys():
                    absolute_score_metric2preds.setdefault(name, [])
                    absolute_score_metric2refs.setdefault(name, [])
                    preds = batch_absolute_score_metric2preds[name].detach().cpu().tolist()
                    refs = [item[1][name] for item in batch["absolute_score_items"]]
                    uids_ref = [batch["uids"][item[0]] for item in batch["absolute_score_items"]]
                    system_ids_ref = [batch["system_ids"][item[0]] for item in batch["absolute_score_items"]]
                    for pred, ref, uid, system_id in zip(preds, refs, uids_ref, system_ids_ref):
                        if uid is None:
                            breakpoint()
                        absolute_score_metric2preds[name].append({"uid": uid, "system_id": system_id, "value": pred})
                        absolute_score_metric2refs[name].append({"uid": uid, "system_id": system_id, "value": ref})

            if "comparative_score" in others:
                others_ = others["comparative_score"]
                for metric in others_["metric2total_items"].keys():
                    total_items = others_["metric2total_items"][metric]
                    correct_items_regression = others_["metric2correct_items_regression"][metric]
                    correct_items_classification = others_["metric2correct_items_classification"][metric]
                    if metric not in comparative_score_metric2info:
                        comparative_score_metric2info[metric] = {
                            "total_items": 0,
                            "correct_items_regression": 0,
                            "correct_items_classification": 0,
                        }
                    comparative_score_metric2info[metric]["total_items"] += total_items
                    comparative_score_metric2info[metric]["correct_items_regression"] += correct_items_regression
                    comparative_score_metric2info[metric][
                        "correct_items_classification"
                    ] += correct_items_classification

        self.accelerator.wait_for_everyone()
        info = {}

        # Sync metric names (accelerate gather_object: all_gather + flatten for lists)
        local_absolute_names = list(absolute_score_metric2preds.keys())
        all_absolute_names = sorted(set(gather_object(local_absolute_names)))

        # Gather absolute score preds/refs with accelerate; compute correlation on main process
        for name in all_absolute_names:
            preds_to_send = absolute_score_metric2preds.get(name, [])
            refs_to_send = absolute_score_metric2refs.get(name, [])
            flat_preds = gather_object(preds_to_send)
            flat_refs = gather_object(refs_to_send)
            absolute_score_metric2preds[name] = flat_preds
            absolute_score_metric2refs[name] = flat_refs

        if self.accelerator.is_main_process and all_absolute_names:
            for name in all_absolute_names:
                preds = absolute_score_metric2preds[name]
                refs = absolute_score_metric2refs[name]
                corr = calculate_metrics(preds, refs)
                for mode in ["utt", "sys"]:
                    for key, value in corr[mode].items():
                        info[f"absolute_score_corr_{mode}_{key}_{name}"] = value

        # Gather comparative score info (pass as single-element list so gather_object returns list of dicts)
        gathered_comparative = gather_object([comparative_score_metric2info])
        merged_comparative = {}
        for d in gathered_comparative:
            for metric, counts in d.items():
                merged_comparative.setdefault(
                    metric,
                    {"total_items": 0, "correct_items_regression": 0, "correct_items_classification": 0},
                )
                merged_comparative[metric]["total_items"] += counts["total_items"]
                merged_comparative[metric]["correct_items_regression"] += counts["correct_items_regression"]
                merged_comparative[metric]["correct_items_classification"] += counts["correct_items_classification"]

        for metric, counts in merged_comparative.items():
            total = counts["total_items"]
            info[f"comparative_score_acc_regression_{metric}"] = counts["correct_items_regression"] / total
            info[f"comparative_score_acc_classification_{metric}"] = counts["correct_items_classification"] / total

        info["loss"] = loss.detach().cpu().item()

        return loss.detach(), info
