# train.py

import os
import sys
import torch
import argparse
from datasets import load_from_disk
from unsloth import FastModel
from trl import SFTTrainer, SFTConfig
from huggingface_hub import snapshot_download

from config_loader import load_config
import audio_tokenizer as audio_tok_module
from audio_tokenizer import formatting_audio_func, initialize_tokenizer


def setup_sparktts_repo(config):
    """Clone Spark-TTS repository if needed and add to path."""
    sparktts_path = config.system["sparktts_path"]

    if not os.path.exists(sparktts_path):
        if config.system["auto_clone_sparktts"]:
            repo_url = config.system["sparktts_repo_url"]
            print(f"Cloning Spark-TTS repository from {repo_url}...")
            os.system(f"git clone {repo_url}")
        else:
            raise FileNotFoundError(
                f"Spark-TTS repository not found at '{sparktts_path}'. "
                "Set 'auto_clone_sparktts: true' in config or clone manually."
            )

    sys.path.append(sparktts_path)


def download_model_if_needed(config):
    """Download pre-trained model from Hugging Face if it doesn't exist."""
    model_dir = config.model["base_model_path"]

    if not os.path.exists(model_dir) and config.model["download_if_missing"]:
        print("Downloading pre-trained Spark-TTS model...")
        snapshot_download(
            config.model["huggingface_model_id"],
            local_dir=model_dir,
            max_workers=config.model["max_workers"]
        )
        print(f"Model downloaded to: {model_dir}")


def load_model_and_tokenizer(config):
    """Load the model and tokenizer with configuration."""
    # Determine which model path to use
    checkpoint_path = config.model.get("checkpoint_path")
    if checkpoint_path and os.path.exists(checkpoint_path):
        model_name = checkpoint_path
        print(f"Loading model from checkpoint: {checkpoint_path}")
    else:
        model_name = config.model["llm_path"]
        print(f"Loading base model from: {model_name}")

    # Parse dtype
    dtype_str = config.model["dtype"]
    dtype_map = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16
    }
    dtype = dtype_map.get(dtype_str, torch.float32)

    model, tokenizer = FastModel.from_pretrained(
        model_name=model_name,
        max_seq_length=config.model["max_seq_length"],
        dtype=dtype,
        full_finetuning=config.model["full_finetuning"],
        load_in_4bit=config.model["load_in_4bit"],
    )

    return model, tokenizer


def apply_lora(model, config):
    """Apply LoRA configuration to the model."""
    if not config.lora["enabled"]:
        print("LoRA disabled in configuration.")
        return model

    print("Applying LoRA configuration...")
    model = FastModel.get_peft_model(
        model,
        r=config.lora["r"],
        target_modules=config.lora["target_modules"],
        lora_alpha=config.lora["lora_alpha"],
        lora_dropout=config.lora["lora_dropout"],
        bias=config.lora["bias"],
        use_gradient_checkpointing=config.lora["use_gradient_checkpointing"],
        random_state=config.lora["random_state"],
        use_rslora=config.lora["use_rslora"],
        loftq_config=config.lora["loftq_config"],
    )

    return model


def load_and_tokenize_dataset(config):
    """Load preprocessed dataset and apply audio tokenization."""
    dataset_path = config.data["preprocessed_dataset_path"]

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Processed dataset not found at '{dataset_path}'. "
            "Please run 'python preprocess_data.py' first."
        )

    print(f"Loading dataset from: {dataset_path}")
    dataset = load_from_disk(dataset_path)

    # Initialize audio tokenizer
    initialize_tokenizer(config)

    # Apply tokenization
    print("Tokenizing audio data...")
    remove_cols = config.training["remove_columns"]
    dataset = dataset.map(formatting_audio_func, remove_columns=remove_cols)

    # Clean up GPU memory if configured
    if config.system["move_to_cpu_after_tokenization"]:
        print("Moving audio models to CPU to free GPU memory...")
        audio_tokenizer = audio_tok_module.audio_tokenizer
        if audio_tokenizer is None:
            print("ERROR: audio_tokenizer is None!")
        elif not hasattr(audio_tokenizer, 'model'):
            print(f"ERROR: audio_tokenizer has no 'model' attribute. Type: {type(audio_tokenizer)}")
            print(f"Available attributes: {dir(audio_tokenizer)}")
        elif audio_tokenizer.model is None:
            print("ERROR: audio_tokenizer.model is None!")
        else:
            audio_tokenizer.model.cpu()
            audio_tokenizer.feature_extractor.cpu()

    if config.system["clear_cuda_cache"]:
        torch.cuda.empty_cache()

    return dataset


def create_trainer(model, tokenizer, dataset, config):
    """Create and configure the SFTTrainer."""
    # Determine training duration
    num_train_epochs = config.training.get("num_train_epochs")
    max_steps = config.training.get("max_steps")

    training_args = {
        "per_device_train_batch_size": config.training["per_device_train_batch_size"],
        "gradient_accumulation_steps": config.training["gradient_accumulation_steps"],
        "warmup_steps": config.training["warmup_steps"],
        "learning_rate": config.training["learning_rate"],
        "fp16": config.training["fp16"],
        "bf16": config.training["bf16"],
        "logging_steps": config.training["logging_steps"],
        "optim": config.training["optim"],
        "weight_decay": config.training["weight_decay"],
        "lr_scheduler_type": config.training["lr_scheduler_type"],
        "seed": config.training["seed"],
        "output_dir": config.training["output_dir"],
        "report_to": config.training["report_to"],
    }

    # Add num_train_epochs or max_steps
    if num_train_epochs is not None:
        training_args["num_train_epochs"] = num_train_epochs
    if max_steps is not None:
        training_args["max_steps"] = max_steps

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field=config.training["dataset_text_field"],
        max_seq_length=config.model["max_seq_length"],
        packing=config.training["packing"],
        args=SFTConfig(**training_args),
    )

    return trainer


def train_spark_tts(config_path="config.yaml"):
    """
    Main training function that orchestrates the entire training process.

    Args:
        config_path: Path to the YAML configuration file.
    """
    # Load configuration
    print(f"Loading configuration from: {config_path}")
    config = load_config(config_path)

    # Setup environment
    setup_sparktts_repo(config)
    download_model_if_needed(config)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config)

    # Apply LoRA
    model = apply_lora(model, config)

    # Load and tokenize dataset
    dataset = load_and_tokenize_dataset(config)

    # Create trainer
    trainer = create_trainer(model, tokenizer, dataset, config)

    # Train
    print("✅ Setup complete. Starting training...")
    trainer.train()
    print("🏁 Training finished.")

    # Save final model
    final_model_path = config.training["final_checkpoint_dir"]
    print(f"Saving the fine-tuned model to {final_model_path}...")
    os.makedirs(final_model_path, exist_ok=True)
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print("Model saved successfully.")


def main():
    """Main function to handle command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Spark-TTS model with configurable parameters."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the configuration YAML file (default: config.yaml)"
    )

    args = parser.parse_args()

    # Run training
    train_spark_tts(args.config)


if __name__ == "__main__":
    main()
