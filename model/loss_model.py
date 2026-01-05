import gc
from dataclasses import dataclass, field
from functools import partial
from typing import Dict

import datasets
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import seed_worker

from loss import FocalLoss, GDiceLoss, TopologicalLoss


class ImprovedLossTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        """
        Initializes the Trainer. We instantiate the loss functions here
        so they are not recreated on every training step, which is more efficient.
        """
        super().__init__(*args, **kwargs)

        # Instantiate all required loss functions once.
        # The specific losses to be used for a given task will be
        # determined by the training arguments inside compute_loss.
        self.topological_loss_fn = TopologicalLoss()
        self.ce_loss_fn = torch.nn.CrossEntropyLoss()
        self.focal_loss_fn = FocalLoss(gamma=2, reduction="mean")

    def training_step(self, model, inputs, num_items_in_batch):
        out = super().training_step(model, inputs, num_items_in_batch)
        if self.state.global_step % self.args.logging_steps == 0:
            gc.collect()
            torch.cuda.empty_cache()
        return out

    def compute_loss(self, model, inputs, num_items_in_batch=1, return_outputs=False):
        labels = inputs["labels"]
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        out = model(
            input_ids=input_ids,
            attention_mask=(
                attention_mask
                if "falcon" not in self.model.base_model.model.name_or_path
                else None
            ),
            labels=labels,
        )
        logits = out.logits
        # Get the most likely token ID at each position by finding the max logit
        predicted_ids = torch.argmax(logits, dim=-1)

        # Shift so that tokens < n predict n
        logits = logits[..., :-1, :].contiguous()
        labels = labels[..., 1:].contiguous()
        # 3. --- Prepare Masks for Targeted Losses ---
        # Create a binary mask for the ingredients section using the pre-calculated token positions.
        ing_start_tokens = inputs["ingredients_start_token"]
        ing_end_tokens = inputs["ingredients_end_token"]
        ingredients_mask = torch.zeros_like(labels)

        for i in range(labels.size(0)):
            start_idx = ing_start_tokens[i].item()
            # The end token from the data script is inclusive, so we add 1 for Python slicing.
            end_idx = ing_end_tokens[i].item() + 1

            # Ensure indices are valid before creating the mask slice.
            if start_idx != -1 and end_idx > start_idx:
                ingredients_mask[i, start_idx:end_idx] = 1

        # Prepare labels for non-cross-entropy losses
        # Set the labels to -100 before inputs["labels_position_id"] for each sample
        labels_start = inputs["labels_position_id"].min()
        # We take from the start of the labels to the end of the sequence (not including the last token)
        processed_labels = labels[:, labels_start - 1 : -1].clone()
        for i, pos in enumerate(inputs["labels_position_id"]):
            processed_labels[i, : pos - labels_start] = -100
        processed_logits = logits[:, labels_start - 1 : -1].contiguous()

        task_losses = self.args.losses["question_answering"]

        losses = {}
        if "cross_entropy" in task_losses:
            lm_loss = self.ce_loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            losses["cross_entropy"] = lm_loss

        if "gdice" in task_losses:
            dice_fct = GDiceLoss(apply_nonlin=torch.nn.Softmax(dim=1))
            dice_loss = dice_fct(processed_logits, processed_labels)
            losses["gdice"] = dice_loss

        if "focal" in task_losses:
            focal_fct = FocalLoss(gamma=2, reduction="mean")
            focal_loss = focal_fct(
                processed_logits.reshape(-1, logits.size(-1)),
                processed_labels.reshape(-1),
            )
            losses["focal"] = focal_loss

        if "focal_ingredients" in task_losses:
            # Use the single, shared ingredients_mask
            active_positions = ingredients_mask == 1
            masked_logits = logits[active_positions]
            masked_labels = labels[active_positions]

            if masked_labels.numel() > 0:
                focal_loss = self.focal_loss_fn(masked_logits, masked_labels)
                losses["focal_ingredients"] = focal_loss
            else:
                # Add a zero loss to prevent a KeyError during the final summation
                losses["focal_ingredients"] = torch.tensor(0.0, device=logits.device)

        if "topological" in task_losses:
            embedding_matrix = model.get_input_embeddings().weight

            # Call the instantiated loss class with the appropriate inputs.
            topo_loss = self.topological_loss_fn(
                logits, labels, ingredients_mask, embedding_matrix
            )
            losses["topological"] = topo_loss
        loss = sum([losses[k] * w for k, w in task_losses.items()])
        return loss if not return_outputs else (loss, out)

    def get_train_dataloader(self) -> DataLoader:
        train_dataloaders = {
            k: self.get_single_dataloader(v) for k, v in self.train_dataset.items()
        }
        if len(train_dataloaders.keys()) != 1:
            raise ValueError("Training multiple dataloader is not supported")
        return list(train_dataloaders.values())[0]

    def get_eval_dataloader(self, eval_dataset: Dataset | None = None) -> DataLoader:
        eval_dataloaders = {
            k: self.get_single_dataloader(v) for k, v in self.eval_dataset.items()
        }
        if len(eval_dataloaders.keys()) != 1:
            raise ValueError("Training multiple dataloader is not supported")
        return list(eval_dataloaders.values())[0]

    def get_single_dataloader(self, dataset) -> DataLoader:
        data_collator = self.data_collator
        if isinstance(dataset, datasets.Dataset):
            dataset = self._remove_unused_columns(dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(
                data_collator, description="training"
            )

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

        if not isinstance(dataset, IterableDataset):
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            worker_init_fn = partial(
                seed_worker,
                num_workers=self.args.dataloader_num_workers,
                rank=self.args.process_index,
            )
            dataloader_params["worker_init_fn"] = worker_init_fn

        return DataLoader(dataset, **dataloader_params)


@dataclass
class ImprovedLossTrainingArgs(TrainingArguments):
    losses: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {"summarization": {"cross_entropy": 1.0}},
        metadata={"help": "losses to use and respective weights"},
    )
