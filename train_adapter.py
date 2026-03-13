#!/usr/bin/env python3
"""
LoRA Adapter Training Script for CodeLLaMA-7B-Python
Trains a LoRA adapter on your custom training dataset.

RunPod paths:
  Dataset:     /workspace/dataset/train.parquet, /workspace/dataset/validation.parquet
  Output:      /workspace/output/codellama-7b-python-adapter
  HF Cache:    /workspace/huggingface_cache (set HF_HOME env var)
"""

import copy
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
from functools import partial

import torch
import transformers
import evaluate
import numpy as np
from torch.utils.data import Dataset
from transformers import (
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    TrainingArguments as HFTrainingArguments,
)
from datasets import load_dataset
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

IGNORE_INDEX = -100

# LoRA Configuration
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj",
    "v_proj",
]


@dataclass
class ModelArguments:
    """Model configuration arguments."""
    model_name_or_path: Optional[str] = field(
        default="codellama/CodeLlama-7b-Python-hf",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    torch_dtype: torch.dtype = field(default=torch.float16)
    device_map: str = field(default="auto")
    use_8bit_quantization: bool = field(
        default=True,
        metadata={"help": "Use 8-bit quantization (recommended for GPUs with <20GB VRAM)"}
    )


@dataclass
class DataArguments:
    """Data configuration arguments."""
    train_data_path: str = field(
        default="/workspace/dataset/train.parquet",
        metadata={"help": "Path to training data file (parquet or jsonl)"}
    )
    validation_data_path: str = field(
        default="/workspace/dataset/validation.parquet",
        metadata={"help": "Path to validation data file (parquet or jsonl)"}
    )


@dataclass
class CustomTrainingArguments(HFTrainingArguments):
    """Custom training arguments."""
    output_dir: str = field(default="/workspace/output/codellama-7b-python-adapter")
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=1024)
    num_train_epochs: float = field(default=3.0)
    per_device_train_batch_size: int = field(default=4)
    per_device_eval_batch_size: int = field(default=4)
    learning_rate: float = field(default=5e-4)
    lr_scheduler_type: str = field(default="cosine")
    logging_steps: int = field(default=50)
    evaluation_strategy: str = field(default="steps")
    eval_steps: int = field(default=500)
    save_strategy: str = field(default="steps")
    save_steps: int = field(default=500)
    save_total_limit: int = field(default=5)
    report_to: str = field(default="tensorboard")
    fp16: bool = field(default=True)
    gradient_accumulation_steps: int = field(default=4)
    warmup_ratio: float = field(default=0.03)
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "Path to checkpoint folder to resume from, or 'True' to auto-detect latest checkpoint in output_dir"}
    )


def tokenize(text, tokenizer, model_max_length, add_eos_token=True):
    """Tokenize text input."""
    result = tokenizer(
        text,
        truncation=True,
        max_length=model_max_length,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < model_max_length
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    if add_eos_token and len(result["input_ids"]) >= model_max_length:
        result["input_ids"][model_max_length - 1] = tokenizer.eos_token_id
        result["attention_mask"][model_max_length - 1] = 1

    result["labels"] = result["input_ids"].copy()
    return result


def get_prompt_target(sample):
    """Extract input and output from sample."""
    return sample['input'], sample['output']


def generate_and_tokenize_prompt(sample, tokenizer, model_max_length):
    """Generate and tokenize prompt-target pairs."""
    input_text, target = get_prompt_target(sample)
    full_text = input_text + target + tokenizer.eos_token

    tokenized_full_text = tokenize(full_text, tokenizer, model_max_length)
    tokenized_input_text = tokenize(input_text, tokenizer, model_max_length)

    input_len = len(tokenized_input_text["input_ids"])
    tokenized_full_text["labels"] = [-100] * input_len + tokenized_full_text["labels"][input_len:]

    return tokenized_full_text


def get_data_module(tokenizer, training_args, data_args) -> Dict:
    """Load and prepare the data module."""
    logger.info(f"Loading training data from {data_args.train_data_path}")
    logger.info(f"Loading validation data from {data_args.validation_data_path}")

    # Load datasets from local parquet files
    train_dataset = load_dataset(
        "parquet",
        data_files=data_args.train_data_path,
        split="train"
    )

    eval_dataset = load_dataset(
        "parquet",
        data_files=data_args.validation_data_path,
        split="train"
    )

    logger.info(f"Loaded {len(train_dataset)} training samples")
    logger.info(f"Loaded {len(eval_dataset)} validation samples")

    # Use available CPU cores for tokenization
    num_workers = min(os.cpu_count() or 2, 8)

    # Tokenize datasets
    train_dataset = train_dataset.map(
        partial(generate_and_tokenize_prompt, tokenizer=tokenizer, model_max_length=training_args.model_max_length),
        num_proc=num_workers,
        desc="Tokenizing training data"
    )

    eval_dataset = eval_dataset.map(
        partial(generate_and_tokenize_prompt, tokenizer=tokenizer, model_max_length=training_args.model_max_length),
        num_proc=num_workers,
        desc="Tokenizing validation data"
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        pad_to_multiple_of=8,
        return_tensors="pt",
        padding=True
    )

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)


def train():
    """Main training function."""
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logger.info("=" * 80)
    logger.info("Model Arguments:")
    logger.info(model_args)
    logger.info("\nData Arguments:")
    logger.info(data_args)
    logger.info("\nTraining Arguments:")
    logger.info(training_args)
    logger.info("=" * 80)

    # Load model
    logger.info(f"Loading model from {model_args.model_name_or_path}")

    # Configure quantization if enabled
    quantization_config = None
    if model_args.use_8bit_quantization:
        logger.info("Using 8-bit quantization")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=model_args.torch_dtype if not model_args.use_8bit_quantization else torch.float16,
        device_map=model_args.device_map,
        trust_remote_code=True,
        quantization_config=quantization_config,
        attn_implementation="flash_attention_2",
    )

    # Prepare model for LoRA training
    if model_args.use_8bit_quantization:
        model = prepare_model_for_kbit_training(model)

    # Configure LoRA
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # Must be after get_peft_model for gradient checkpointing to work
    if not model_args.use_8bit_quantization:
        model.enable_input_require_grads()

    logger.info("LoRA Configuration:")
    logger.info(f"  r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}")
    logger.info(f"  Target modules: {LORA_TARGET_MODULES}")
    if model_args.use_8bit_quantization:
        logger.info("  Quantization: 8-bit (memory efficient)")
    model.print_trainable_parameters()

    # Load tokenizer
    logger.info(f"Loading tokenizer from {model_args.model_name_or_path}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="left",
        use_fast=True,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Resize embeddings to match tokenizer (CodeLLaMA tokenizer has 32005 tokens)
    model.resize_token_embeddings(len(tokenizer))

    # Get data module
    data_module = get_data_module(tokenizer=tokenizer, training_args=training_args, data_args=data_args)

    # Initialize trainer
    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        **data_module,
    )

    # Train (with optional checkpoint resume)
    resume_checkpoint = None
    if training_args.resume_from_checkpoint:
        if training_args.resume_from_checkpoint.lower() == "true":
            # Auto-detect latest checkpoint in output_dir
            import glob
            checkpoints = sorted(glob.glob(os.path.join(training_args.output_dir, "checkpoint-*")))
            if checkpoints:
                resume_checkpoint = checkpoints[-1]
                logger.info(f"Auto-detected checkpoint: {resume_checkpoint}")
            else:
                logger.info("No checkpoints found in output_dir, starting fresh")
        else:
            resume_checkpoint = training_args.resume_from_checkpoint
            logger.info(f"Resuming from checkpoint: {resume_checkpoint}")

    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    # Save
    logger.info(f"Saving adapter to {training_args.output_dir}")
    trainer.save_state()
    model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

    logger.info("Training complete!")


if __name__ == "__main__":
    train()
