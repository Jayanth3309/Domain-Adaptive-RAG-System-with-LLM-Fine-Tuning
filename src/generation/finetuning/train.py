"""
QLoRA fine-tuning script for Mistral-7B on domain-specific QA pairs.
Uses PEFT + bitsandbytes 4-bit quantization.
Logs all experiments to MLflow.
"""

import os
import json
import argparse
from pathlib import Path

import torch
import mlflow
import yaml
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from trl import SFTTrainer
from loguru import logger


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_dataset_from_jsonl(path: str) -> Dataset:
    """Load QA pairs from JSONL. Expected keys: instruction, input, output."""
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    logger.info(f"Loaded {len(records)} training samples from {path}")
    return Dataset.from_list(records)


def format_prompt(example: dict) -> dict:
    """Format into Mistral instruction template."""
    prompt = (
        f"<s>[INST] {example['instruction']}\n\n"
        f"Context: {example['input']} [/INST] "
        f"{example['output']} </s>"
    )
    return {"text": prompt}


def build_model_and_tokenizer(config: dict):
    model_name = config["model"]["base_model"]

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config["quantization"]["load_in_4bit"],
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type=config["quantization"]["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=config["quantization"]["bnb_4bit_use_double_quant"],
    )

    logger.info(f"Loading base model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    lora_cfg = config["qlora"]
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg["bias"],
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer, peft_config


def train(config_path: str, train_file: str, eval_file: str):
    config = load_config(config_path)
    train_cfg = config["training"]

    # MLflow experiment
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name="qlora-finetune"):
        mlflow.log_params({
            "base_model": config["model"]["base_model"],
            "lora_r": config["qlora"]["r"],
            "lora_alpha": config["qlora"]["lora_alpha"],
            "epochs": train_cfg["num_train_epochs"],
            "lr": train_cfg["learning_rate"],
            "batch_size": train_cfg["per_device_train_batch_size"],
        })

        model, tokenizer, peft_config = build_model_and_tokenizer(config)

        train_dataset = load_dataset_from_jsonl(train_file)
        eval_dataset = load_dataset_from_jsonl(eval_file)

        train_dataset = train_dataset.map(format_prompt)
        eval_dataset = eval_dataset.map(format_prompt)

        training_args = TrainingArguments(
            output_dir=train_cfg["output_dir"],
            num_train_epochs=train_cfg["num_train_epochs"],
            per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
            gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
            learning_rate=train_cfg["learning_rate"],
            lr_scheduler_type=train_cfg["lr_scheduler_type"],
            warmup_ratio=train_cfg["warmup_ratio"],
            max_grad_norm=train_cfg["max_grad_norm"],
            fp16=train_cfg["fp16"],
            logging_steps=train_cfg["logging_steps"],
            save_steps=train_cfg["save_steps"],
            eval_steps=train_cfg["eval_steps"],
            evaluation_strategy="steps",
            save_total_limit=train_cfg["save_total_limit"],
            report_to="none",  # handled by MLflow manually
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
        )

        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=peft_config,
            dataset_text_field="text",
            tokenizer=tokenizer,
            args=training_args,
            max_seq_length=2048,
        )

        logger.info("Starting QLoRA fine-tuning...")
        train_result = trainer.train()

        # Log final metrics
        mlflow.log_metrics({
            "train_loss": train_result.training_loss,
            "train_runtime_s": train_result.metrics["train_runtime"],
        })

        trainer.save_model(train_cfg["output_dir"])
        tokenizer.save_pretrained(train_cfg["output_dir"])
        logger.info(f"Model saved to {train_cfg['output_dir']}")

        mlflow.log_artifact(train_cfg["output_dir"])
        logger.info("Fine-tuning complete. Run logged to MLflow.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--train-file", default="data/processed/train_qa.jsonl")
    parser.add_argument("--eval-file", default="data/processed/eval_qa.jsonl")
    args = parser.parse_args()
    train(args.config, args.train_file, args.eval_file)
