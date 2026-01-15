#!/usr/bin/env python3
"""
Environment check script for Spark-TTS Fine-tuning.
Verifies if all dependencies and configurations are correct.
"""

import sys
import os
from pathlib import Path


def print_header(text):
    """Print formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_check(name, status, details=""):
    """Print check result."""
    status_symbol = "✓" if status else "✗"
    status_text = "OK" if status else "ERRO"
    print(f"[{status_symbol}] {name}: {status_text}")
    if details:
        print(f"    {details}")


def check_python_version():
    """Check Python version."""
    print_header("Checking Python Version")
    version = sys.version_info
    is_valid = version.major == 3 and version.minor >= 8
    print_check(
        "Python Version",
        is_valid,
        f"Python {version.major}.{version.minor}.{version.micro} "
        f"({'OK' if is_valid else 'Requires Python 3.8+'})"
    )
    return is_valid


def check_pytorch():
    """Check PyTorch installation and CUDA availability."""
    print_header("Checking PyTorch and CUDA")

    try:
        import torch
        print_check("PyTorch installed", True, f"Version: {torch.__version__}")

        cuda_available = torch.cuda.is_available()
        print_check("CUDA available", cuda_available)

        if cuda_available:
            print(f"    GPU: {torch.cuda.get_device_name(0)}")
            print(f"    CUDA Version: {torch.version.cuda}")
            print(f"    GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

        return True
    except ImportError:
        print_check("PyTorch installed", False, "Run: pip install torch")
        return False


def check_dependencies():
    """Check required dependencies."""
    print_header("Checking Dependencies")

    dependencies = {
        "transformers": "Hugging Face Transformers",
        "datasets": "Hugging Face Datasets",
        "unsloth": "Unsloth",
        "trl": "TRL (Transformer Reinforcement Learning)",
        "soundfile": "SoundFile",
        "torchaudio": "TorchAudio",
        "yaml": "PyYAML",
        "pandas": "Pandas",
        "tqdm": "TQDM"
    }

    all_ok = True
    for module, name in dependencies.items():
        try:
            __import__(module)
            print_check(name, True)
        except ImportError:
            print_check(name, False, f"Run: pip install {module}")
            all_ok = False

    return all_ok


def check_config_file():
    """Check if config file exists and is valid."""
    print_header("Checking Configuration File")

    config_path = Path("config.yaml")
    config_exists = config_path.exists()
    print_check("config.yaml exists", config_exists)

    if not config_exists:
        print("    Create config.yaml from config.example.yaml")
        return False

    try:
        from config_loader import load_config
        config = load_config("config.yaml")
        print_check("config.yaml is valid", True)

        # Check critical paths
        print("\n  Configured Paths:")
        print(f"    Dataset: {config.data['local_dataset_path']}")
        print(f"    Output: {config.training['output_dir']}")
        print(f"    Model: {config.model['base_model_path']}")

        return True
    except Exception as e:
        print_check("config.yaml is valid", False, str(e))
        return False


def check_dataset():
    """Check if dataset is properly configured."""
    print_header("Checking Dataset")

    try:
        from config_loader import load_config
        config = load_config("config.yaml")

        dataset_path = Path(config.data["local_dataset_path"])
        dataset_exists = dataset_path.exists()
        print_check("Dataset directory exists", dataset_exists, str(dataset_path))

        if dataset_exists:
            metadata_path = dataset_path / "metadata.csv"
            metadata_exists = metadata_path.exists()
            print_check("metadata.csv exists", metadata_exists, str(metadata_path))

            if metadata_exists:
                import pandas as pd
                try:
                    df = pd.read_csv(
                        metadata_path,
                        sep=config.data["metadata"]["separator"],
                        header=config.data["metadata"]["header"],
                        nrows=5
                    )
                    print_check("metadata.csv is readable", True, f"{len(df)} lines (sample)")
                    print(f"    Columns: {df.columns.tolist()}")
                    return True
                except Exception as e:
                    print_check("metadata.csv is readable", False, str(e))
                    return False

        return False
    except Exception as e:
        print(f"    Error checking dataset: {e}")
        return False


def check_preprocessed_data():
    """Check if data has been preprocessed."""
    print_header("Checking Preprocessed Data")

    try:
        from config_loader import load_config
        config = load_config("config.yaml")

        preprocessed_path = Path(config.data["preprocessed_dataset_path"])
        preprocessed_exists = preprocessed_path.exists()
        print_check(
            "Preprocessed dataset exists",
            preprocessed_exists,
            str(preprocessed_path) if preprocessed_exists else "Run: python preprocess_data.py"
        )

        if preprocessed_exists:
            # Check if it's a valid dataset
            try:
                from datasets import load_from_disk
                dataset = load_from_disk(str(preprocessed_path))
                print(f"    Dataset size: {len(dataset)} examples")
                print(f"    Columns: {dataset.column_names}")
                return True
            except Exception as e:
                print_check("Preprocessed dataset is valid", False, str(e))
                return False

        return False
    except Exception as e:
        print(f"    Error checking preprocessed data: {e}")
        return False


def check_spark_tts_repo():
    """Check if Spark-TTS repository exists."""
    print_header("Checking Spark-TTS Repository")

    spark_tts_path = Path("Spark-TTS")
    exists = spark_tts_path.exists()
    print_check(
        "Spark-TTS Repository",
        exists,
        "Will be cloned automatically when running scripts" if not exists else str(spark_tts_path)
    )
    return True  # Not critical, will be cloned automatically


def check_disk_space():
    """Check available disk space."""
    print_header("Checking Disk Space")

    try:
        import shutil
        total, used, free = shutil.disk_usage(".")
        free_gb = free / (1024 ** 3)

        sufficient = free_gb > 10
        print_check(
            "Free disk space",
            sufficient,
            f"{free_gb:.2f} GB available ({'Sufficient' if sufficient else 'Recommended >10GB'})"
        )
        return sufficient
    except Exception as e:
        print(f"    Unable to check disk space: {e}")
        return True


def main():
    """Run all checks."""
    print("\n" + "=" * 70)
    print("  ENVIRONMENT CHECK - Spark-TTS Fine-tuning")
    print("=" * 70)

    results = {
        "Python": check_python_version(),
        "PyTorch": check_pytorch(),
        "Dependencies": check_dependencies(),
        "Config": check_config_file(),
        "Dataset": check_dataset(),
        "Preprocessed": check_preprocessed_data(),
        "Spark-TTS": check_spark_tts_repo(),
        "Disk Space": check_disk_space()
    }

    # Summary
    print_header("SUMMARY")
    passed = sum(results.values())
    total = len(results)

    for name, status in results.items():
        symbol = "✓" if status else "✗"
        print(f"  {symbol} {name}")

    print(f"\n  Total: {passed}/{total} checks passed")

    if passed == total:
        print("\n  ✓ Environment is ready to use!")
        print("  Run: python preprocess_data.py  # If not preprocessed yet")
        print("       python train.py             # To train")
        sys.exit(0)
    else:
        print("\n  ✗ Some checks failed. Fix the issues above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
