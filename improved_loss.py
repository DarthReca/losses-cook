import logging
import os
import re
from datetime import datetime
from typing import Optional

import comet_ml
import hydra
import numpy as np
import pandas as pd
import torch
import transformers
from datasets import DatasetInfo, load_dataset
from datasets.features import Sequence, Value
from huggingface_hub import login
from omegaconf import DictConfig
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import DataCollatorForLanguageModeling
from transformers.integrations import CometCallback

from model import ImprovedLossTrainer, ImprovedLossTrainingArgs

login(token="")


class CometCallBackWithName(CometCallback):
    def __init__(self, experiment_name: str):
        self._initialized = False
        self._log_assets = False
        self.experiment_name = experiment_name

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        super().on_train_begin(args, state, control, model, **kwargs)
        comet_ml.config.get_global_experiment().set_name(self.experiment_name)


class CustomDataCollator(DataCollatorForLanguageModeling):
    def __call__(self, features, return_tensors=None):
        # Extract and remove your custom columns that are not tensors
        # This prevents the default collator from trying to pad them.
        ner_ingredients_quantity = [
            f.pop("NER_ingredients_quantity", None) for f in features
        ]
        ner_ingredients = [f.pop("NER_ingredients", None) for f in features]
        task_type = [f.pop("task_type", None) for f in features]
        ground_truth_prediction = [f.pop("answer", None) for f in features]

        # Let the parent class handle the standard collating (padding, tensor conversion)
        batch = super().__call__(features, return_tensors)

        # Now, add your custom data back into the batch.
        # It will be a list of lists, not a tensor.
        batch["NER_ingredients_quantity"] = ner_ingredients_quantity
        batch["NER_ingredients"] = ner_ingredients
        batch["task_type"] = task_type
        batch["ground_truth_prediction"] = ground_truth_prediction

        return batch


def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak.
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    pred_ids = torch.argmax(logits[0] if isinstance(logits, tuple) else logits, dim=-1)
    return pred_ids


@hydra.main(config_path="configs", config_name="loss", version_base=None)
def main(args: DictConfig):

    logging.basicConfig(level=logging.INFO)
    transformers.set_seed(42)
    name = f"{args.experiment_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model.model_name, padding_side="left"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    train_dataset = {}
    val_dataset = {}
    df = pd.read_csv(f"./dataset/{args.dataset_source}_ingredients.csv")
    ingredient_list = df["ingredients"].to_list()
    for k, v in args.dataset.train_set_file.items():
        train_dataset[k] = load_dataset("parquet", data_files=v)["train"].cast_column(
            "attention_mask", Sequence(Value("bool"))
        )
        train_dataset[k] = (
            train_dataset[k]
            .add_column("task", [0] * len(train_dataset[k]))
            .filter(lambda x: x["labels_position_id"] != [0])
        )
        train_dataset[k].info.metadata = {"ingredient_list": ingredient_list}
    # let's check if we can access the ingredient list from the metadata
    ingredient_list = train_dataset[k].info.metadata["ingredient_list"]
    maximum_val_size = 100 // len(args.dataset.val_set_file)
    for k, v in args.dataset.val_set_file.items():
        val_dataset[k] = load_dataset("parquet", data_files=v, cache_dir="cache")[
            "train"
        ]
        val_dataset[k] = val_dataset[k].select(
            range(min(maximum_val_size, len(val_dataset[k])))
        )
        val_dataset[k] = val_dataset[k].add_column("task", [0] * len(val_dataset[k]))

    if args.log_comet:
        comet_ml.init()
    ### LORA CONFIG
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model.model_name, cache_dir="cache", trust_remote_code=False
    )
    model.resize_token_embeddings(len(tokenizer))

    training_arguments = ImprovedLossTrainingArgs(
        output_dir=f"checkpoints/{name}",
        remove_unused_columns=False,
        **args.trainer,
    )
    # Use LoRA

    # prepare int-8 model for training
    model = prepare_model_for_kbit_training(model)
    # add LoRA adaptor
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    data_collator = CustomDataCollator(tokenizer=tokenizer, mlm=False)
    trainer = ImprovedLossTrainer(
        model=model,
        # tokenizer=tokenizer,
        processing_class=tokenizer,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[CometCallBackWithName(name)] if args.log_comet else None,
        data_collator=data_collator,
    )

    trainer.train()
    output_dir = f"./finetuned_models/{name}"

    # First, save the final model state. If load_best_model_at_end=True and a best
    # model was found, this will be the best model. If not, it's the last one.
    print(f"Saving final model to {output_dir}")
    trainer.save_model(output_dir)
    # Now, save the tokenizer to that same directory to create a complete, portable package.
    print(f"Saving tokenizer to {output_dir}")
    tokenizer.save_pretrained(output_dir)

    # Optional: Log which checkpoint was considered the best, for your own information.
    if trainer.state.best_model_checkpoint:
        print(
            f"The best model checkpoint during training was at: {trainer.state.best_model_checkpoint}"
        )
    else:
        print(
            "Note: No 'best' model was found based on the validation metric. The final model state from the last step has been saved instead."
        )


if __name__ == "__main__":
    main()
