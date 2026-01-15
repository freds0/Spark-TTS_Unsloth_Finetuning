# Spark-TTS Fine-tuning on LJSpeech

This repository contains scripts to fine-tune the Spark-TTS model on custom datasets for text-to-speech (TTS) synthesis. All scripts have been refactored to use a centralized YAML configuration file for easy customization.

## Project Structure

```
spark-tts-ljspeech/
│
├── data/
│   ├── LJSpeech-1.1/
│   └── preprocessed_dataset/
│
├── models/
│   └── Spark-TTS-0.5B/
│
├── outputs/
│   ├── checkpoint-*/
│   └── synthesis_batch/
│
├── Spark-TTS/                  # Cloned automatically
│
├── config.yaml                 # Main configuration file
├── config_loader.py            # Configuration loader module
├── preprocess_data.py          # Data preprocessing script
├── audio_tokenizer.py          # Audio tokenization module
├── train.py                    # Training script
├── synthesize.py               # Synthesis/inference script
├── requirements.txt
└── README.md
```

## Features

- **Centralized Configuration**: All parameters managed via `config.yaml`
- **Flexible Data Processing**: Support for custom metadata formats
- **LoRA Fine-tuning**: Configurable LoRA parameters for efficient training
- **Batch Synthesis**: Generate multiple audio files efficiently
- **Command-line Overrides**: Override config values via CLI arguments
- **Auto-download Models**: Automatically download pre-trained models from HuggingFace

## Setup

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd spark-tts-ljspeech
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   *Note: Ensure you have a compatible PyTorch version with CUDA support for GPU acceleration.*

3. **Configure your project:**

   Edit the `config.yaml` file to customize paths, hyperparameters, and other settings. The configuration file is well-documented with comments explaining each parameter.

## Configuration File

The `config.yaml` file contains all configuration parameters organized into sections:

### Main Sections

- **data**: Dataset paths, metadata format, audio settings
- **model**: Model paths, dtype, download settings
- **lora**: LoRA configuration (rank, alpha, target modules)
- **training**: Training hyperparameters (batch size, learning rate, epochs)
- **audio_tokenizer**: Audio tokenization settings
- **synthesis**: Inference/synthesis parameters
- **system**: System-level settings (CUDA cache, auto-clone)

### Quick Configuration Tips

1. **Dataset Path**: Set `data.local_dataset_path` to your dataset directory
2. **Training Duration**: Use either `training.num_train_epochs` OR `training.max_steps` (not both)
3. **LoRA Settings**: Adjust `lora.r` (rank) for training efficiency vs. quality trade-off
4. **Synthesis**: Set `synthesis.input_file` to your text file for generation

## Usage

### 1. Data Preprocessing

Preprocess your dataset before training:

```bash
# Using default config.yaml
python preprocess_data.py

# Using a custom config file
python preprocess_data.py --config my_config.yaml

# Force re-processing (overwrite existing)
python preprocess_data.py --force
```

**Dataset Requirements:**
- Your dataset folder should contain a `metadata.csv` file
- Default format: `audio_file|text|speaker_name` (configurable in `config.yaml`)
- Audio files should be in the same directory or use full paths

This script will:
- Load metadata from your dataset
- Resample audio to the configured sampling rate (default: 16kHz)
- Save processed data to `data/preprocessed_dataset` (or configured path)

### 2. Training

Train or fine-tune the Spark-TTS model:

```bash
# Using default config.yaml
python train.py

# Using a custom config file
python train.py --config my_config.yaml
```

The training script will:
- Automatically download the pre-trained model if not present
- Load the preprocessed dataset
- Apply audio tokenization
- Configure LoRA (if enabled)
- Train the model
- Save checkpoints to the output directory

**Key Training Parameters** (edit in `config.yaml`):
- `training.per_device_train_batch_size`: Batch size per GPU
- `training.max_steps`: Number of training steps (or use `num_train_epochs`)
- `training.learning_rate`: Learning rate (default: 2e-4)
- `lora.r`: LoRA rank (higher = more parameters, better quality but slower)

### 3. Synthesis (Text-to-Speech)

Generate speech from text files:

```bash
# Basic usage with config file
python synthesize.py --input_file sentences.txt

# With custom config
python synthesize.py --config my_config.yaml --input_file sentences.txt

# Override specific parameters
python synthesize.py \
  --input_file sentences.txt \
  --model_path outputs/checkpoint-1500 \
  --output_dir my_outputs \
  --batch_size 8
```

**Input File Format:**
- Plain text file with one sentence per line
- UTF-8 encoding

**Example `sentences.txt`:**
```
Hello, welcome to Spark TTS synthesis.
This is a test of the fine-tuned model.
The quick brown fox jumps over the lazy dog.
```

The synthesis script will:
- Load the fine-tuned model
- Process sentences in configurable batches
- Generate `.wav` files in the output directory
- Skip existing files (if `skip_existing: true` in config)

### 4. Configuration Management

You can also work with configurations programmatically:

```python
from config_loader import load_config

# Load configuration
config = load_config("config.yaml")

# Access settings
print(f"Batch size: {config.training['per_device_train_batch_size']}")
print(f"Learning rate: {config.training['learning_rate']}")

# Update and save
config.update("training", "learning_rate", 1e-4)
config.save("new_config.yaml")

# Display specific section
config.display("lora")
```

## Advanced Configuration

### Custom Metadata Format

If your dataset has a different metadata format, adjust in `config.yaml`:

```yaml
data:
  metadata:
    separator: ","  # Change from | to ,
    header: null    # Set to null if no header row
    columns:
      audio_file: "wav_path"      # Your audio column name
      text: "transcription"       # Your text column name
      speaker_name: "speaker_id"  # Your speaker column name
```

### Multi-GPU Training

To use multiple GPUs, use PyTorch's distributed training:

```bash
torchrun --nproc_per_node=2 train.py --config config.yaml
```

### Resume Training from Checkpoint

Set the checkpoint path in `config.yaml`:

```yaml
model:
  checkpoint_path: "./outputs/checkpoint-1500/"
```

Or keep it as `null` to start from the base model.

### Adjust Generation Parameters

Fine-tune synthesis quality in `config.yaml`:

```yaml
synthesis:
  generation:
    temperature: 0.8  # Higher = more random, lower = more deterministic
    top_k: 50         # Consider only top K tokens
    top_p: 1.0        # Nucleus sampling threshold
```

## Common Issues and Solutions

### Out of Memory (OOM) Errors

1. Reduce `training.per_device_train_batch_size`
2. Increase `training.gradient_accumulation_steps`
3. Enable `system.clear_cuda_cache: true`
4. Reduce `lora.r` (LoRA rank)

### Model Not Found

The model will be downloaded automatically if `model.download_if_missing: true`. Ensure you have internet connectivity and sufficient disk space (~2-3 GB).

### Audio Quality Issues

1. Increase `training.max_steps` or `num_train_epochs`
2. Increase `lora.r` for more model capacity
3. Adjust `synthesis.generation.temperature` (try values between 0.6-1.0)
4. Ensure your dataset audio quality is good

## Dependencies

Key dependencies include:
- PyTorch (with CUDA support)
- Transformers
- Datasets
- Unsloth
- TRL (Transformer Reinforcement Learning)
- soundfile
- torchaudio
- PyYAML

See `requirements.txt` for the complete list.

## Citation

If you use Spark-TTS in your research, please cite:

```bibtex
@misc{spark-tts,
  title={Spark-TTS: High-Quality Text-to-Speech Synthesis},
  author={SparkAudio Team},
  year={2024},
  url={https://github.com/SparkAudio/Spark-TTS}
}
```

## License

This project uses the Spark-TTS model, which has its own license. Please refer to the [Spark-TTS repository](https://github.com/SparkAudio/Spark-TTS) for licensing information.

## Support

For issues and questions:
- Check the configuration file for parameter documentation
- Review error messages for missing files or configuration issues
- Ensure all paths in `config.yaml` are correct
- Verify that your dataset format matches the configuration

## Acknowledgments

- [Spark-TTS](https://github.com/SparkAudio/Spark-TTS) for the base model
- [Unsloth](https://github.com/unslothai/unsloth) for efficient fine-tuning
- [Hugging Face](https://huggingface.co) for model hosting and libraries
