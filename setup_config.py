#!/usr/bin/env python3
"""
Interactive script to create a custom config.yaml.
Facilitates the initial project setup.
"""

import os
import shutil
from pathlib import Path


def print_header(text):
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")


def ask_path(prompt, default=None, must_exist=False):
    """Ask for a file path with validation."""
    while True:
        if default:
            user_input = input(f"{prompt} [{default}]: ").strip()
            path = user_input if user_input else default
        else:
            path = input(f"{prompt}: ").strip()

        if not path:
            print("  ⚠ Path cannot be empty.")
            continue

        path_obj = Path(path)

        if must_exist and not path_obj.exists():
            print(f"  ⚠ Path does not exist: {path}")
            create = input("  Do you want to create it? (y/n): ").lower()
            if create == 'y':
                try:
                    path_obj.mkdir(parents=True, exist_ok=True)
                    print(f"  ✓ Directory created: {path}")
                    return path
                except Exception as e:
                    print(f"  ✗ Error creating directory: {e}")
                    continue
            else:
                continue

        return path


def ask_number(prompt, default, min_val=None, max_val=None):
    """Ask for a number with validation."""
    while True:
        user_input = input(f"{prompt} [{default}]: ").strip()
        value = user_input if user_input else str(default)

        try:
            num = float(value)
            if min_val is not None and num < min_val:
                print(f"  ⚠ Value must be >= {min_val}")
                continue
            if max_val is not None and num > max_val:
                print(f"  ⚠ Value must be <= {max_val}")
                continue
            return num
        except ValueError:
            print("  ⚠ Please enter a valid number.")


def ask_choice(prompt, choices, default=None):
    """Ask for a choice from a list."""
    print(f"\n{prompt}")
    for i, choice in enumerate(choices, 1):
        marker = " (default)" if choice == default else ""
        print(f"  {i}. {choice}{marker}")

    while True:
        user_input = input(f"\nChoose (1-{len(choices)}) [{choices.index(default)+1 if default else ''}]: ").strip()

        if not user_input and default:
            return default

        try:
            idx = int(user_input) - 1
            if 0 <= idx < len(choices):
                return choices[idx]
            else:
                print(f"  ⚠ Choose a number between 1 and {len(choices)}")
        except ValueError:
            print("  ⚠ Please enter a valid number.")


def main():
    print_header("INTERACTIVE CONFIGURATOR - Spark-TTS Fine-tuning")

    # Check if config.yaml already exists
    if os.path.exists("config.yaml"):
        print("⚠ config.yaml already exists!")
        overwrite = input("Do you want to overwrite it? (y/n): ").lower()
        if overwrite != 'y':
            print("\n✓ Operation cancelled. config.yaml was not modified.")
            return

    # Check if example exists
    if not os.path.exists("config.example.yaml"):
        print("✗ Error: config.example.yaml not found!")
        print("This file is needed as a template.")
        return

    print("\nLet's configure your project step by step.")
    print("Press Enter to use the default value in brackets.")

    # 1. Dataset
    print_header("1. Dataset Configuration")
    dataset_path = ask_path(
        "Path to dataset directory (containing metadata.csv)",
        default="./LJSpeech-1.1/",
        must_exist=False
    )

    # 2. Training duration
    print_header("2. Training Duration")
    print("\nChoose how to define training duration:")
    duration_choice = ask_choice(
        "Duration method:",
        ["Number of steps", "Number of epochs"],
        default="Number of steps"
    )

    if duration_choice == "Number of steps":
        max_steps = int(ask_number("Number of steps", default=1000, min_val=1))
        num_epochs = None
    else:
        num_epochs = int(ask_number("Number of epochs", default=5, min_val=1))
        max_steps = None

    # 3. Hardware
    print_header("3. Hardware Configuration")
    batch_size = int(ask_number(
        "Batch size per GPU (smaller = less memory)",
        default=2,
        min_val=1,
        max_val=32
    ))

    gradient_accumulation = int(ask_number(
        "Gradient accumulation steps (larger = larger effective batch)",
        default=4,
        min_val=1,
        max_val=64
    ))

    # 4. LoRA
    print_header("4. LoRA Configuration")
    print("\nLoRA Rank (higher = more parameters, better quality, slower)")
    print("  • 32-64: Fast, low memory")
    print("  • 128: Balanced (recommended)")
    print("  • 256+: High quality, slower")
    lora_r = int(ask_number("LoRA rank", default=128, min_val=8, max_val=512))

    # 5. Learning rate
    print_header("5. Hyperparameters")
    lr = ask_number("Learning rate", default=0.0002, min_val=0.00001, max_val=0.01)

    # 6. Synthesis
    print_header("6. Synthesis Configuration")
    temp = ask_number(
        "Temperature (0.6-1.0, lower = more deterministic)",
        default=0.8,
        min_val=0.1,
        max_val=2.0
    )

    synth_batch = int(ask_number(
        "Synthesis batch size",
        default=4,
        min_val=1,
        max_val=32
    ))

    # Create config
    print_header("Creating config.yaml")

    # Copy example and modify
    shutil.copy("config.example.yaml", "config.yaml")

    # Read and modify
    with open("config.yaml", "r") as f:
        content = f.read()

    # Replace values
    replacements = {
        'local_dataset_path: "./seu_dataset/"': f'local_dataset_path: "{dataset_path}"',
        'max_steps: 1000': f'max_steps: {max_steps}' if max_steps else 'max_steps: null',
        'num_train_epochs: null': f'num_train_epochs: {num_epochs}' if num_epochs else 'num_train_epochs: null',
        'per_device_train_batch_size: 2': f'per_device_train_batch_size: {batch_size}',
        'gradient_accumulation_steps: 4': f'gradient_accumulation_steps: {gradient_accumulation}',
        'r: 128  # Rank do LoRA': f'r: {lora_r}',
        'learning_rate: 0.0002': f'learning_rate: {lr}',
        'temperature: 0.8': f'temperature: {temp}',
        'batch_size: 4': f'batch_size: {synth_batch}',
    }

    for old, new in replacements.items():
        content = content.replace(old, new)

    with open("config.yaml", "w") as f:
        f.write(content)

    print("✓ config.yaml created successfully!")

    # Summary
    print_header("CONFIGURATION SUMMARY")
    print(f"Dataset: {dataset_path}")
    if max_steps:
        print(f"Training: {max_steps} steps")
    else:
        print(f"Training: {num_epochs} epochs")
    print(f"Batch size: {batch_size} (effective: {batch_size * gradient_accumulation})")
    print(f"LoRA rank: {lora_r}")
    print(f"Learning rate: {lr}")
    print(f"Temperature (synthesis): {temp}")

    print("\n" + "=" * 70)
    print("✓ Configuration complete!")
    print("\nNext steps:")
    print("  1. Review config.yaml and adjust if necessary")
    print("  2. Run: python check_environment.py")
    print("  3. Run: python preprocess_data.py")
    print("  4. Run: python train.py")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Operation cancelled by user.")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
