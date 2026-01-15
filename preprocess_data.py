# preprocess_data.py

import os
import argparse
import pandas as pd
from datasets import Dataset, Audio
from config_loader import load_config
def preprocess_data_from_local(config):
    """
    Loads and preprocesses a local dataset with a specific metadata format.
    - Expects metadata format: audio_file|text|speaker_name
    - The 'audio_file' column must contain the full path to the wav file.
    - Saves the processed dataset to the configured output directory for faster loading.

    Args:
        config: Config object containing all configuration parameters.
    """
    local_dataset_path = config.data["local_dataset_path"]
    output_dir = config.data["preprocessed_dataset_path"]
    metadata_config = config.data["metadata"]

    metadata_path = os.path.join(local_dataset_path, 'metadata_coqui.csv')

    # Check if metadata file exists
    if not os.path.exists(metadata_path):
        print(f"Error: metadata.csv not found in '{local_dataset_path}'")
        return

    print(f"Loading metadata from: {metadata_path}")

    # Read the metadata file with configured separator and header
    try:
        # Use configured separator and header settings
        metadata_df = pd.read_csv(
            metadata_path,
            sep=metadata_config["separator"],
            header=metadata_config["header"],
            on_bad_lines='skip'
        )

        # Get expected column names from config
        expected_columns = [
            metadata_config["columns"]["audio_file"],
            metadata_config["columns"]["text"],
            metadata_config["columns"]["speaker_name"]
        ]

        # Verify the expected columns are present
        if not all(col in metadata_df.columns for col in expected_columns):
            print(f"Error: metadata.csv is missing one of the expected columns: {expected_columns}")
            print(f"Found columns: {metadata_df.columns.tolist()}")
            return

    except Exception as e:
        print(f"Failed to parse metadata.csv. Error: {e}")
        return

    # Build full paths for audio files
    audio_col = metadata_config["columns"]["audio_file"]
    metadata_df[audio_col] = metadata_df[audio_col].apply(
        lambda x: os.path.basename(x)
    )
    metadata_df[audio_col] = local_dataset_path + "/wavs/" + metadata_df[audio_col]

    # Convert the pandas DataFrame to a Hugging Face Dataset object
    dataset = Dataset.from_pandas(metadata_df)

    print("Dataset loaded successfully from local metadata file.")

    # Cast the audio column to the Audio feature type with configured sampling rate
    target_sr = config.data["audio"]["target_sampling_rate"]
    print(f"Casting audio column to {target_sr}Hz sampling rate...")
    dataset = dataset.cast_column(audio_col, Audio(sampling_rate=target_sr))

    # Rename for consistency with the training script
    dataset = dataset.rename_column(audio_col, "audio")

    # Remove the speaker_name column
    speaker_col = metadata_config["columns"]["speaker_name"]
    if speaker_col in dataset.column_names:
        dataset = dataset.remove_columns([speaker_col])

    # Save the processed dataset in the efficient Arrow format
    print(f"Saving processed dataset to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    dataset.save_to_disk(output_dir)
    print(f"Preprocessing complete. Processed data saved to '{output_dir}'.")
    print("The dataset is now ready for training.")

    
def preprocess_ljspeech(config):
    """
    Loads and preprocesses the LJSpeech dataset.
    - Expects metadata format: filename|text|text_norm (no header)
    - 'filename' should be the relative path to the wav file.
    - Saves the processed dataset to the configured output directory.

    Args:
        config: Config object containing all configuration parameters.
    """
    local_dataset_path = config.data["local_dataset_path"]
    output_dir = config.data["preprocessed_dataset_path"]
    metadata_config = config.data["metadata"]

    metadata_path = os.path.join(local_dataset_path, 'metadata.csv')

    # Check if metadata file exists
    if not os.path.exists(metadata_path):
        print(f"Error: metadata.csv not found in '{local_dataset_path}'")
        return

    print(f"Loading LJSpeech metadata from: {metadata_path}")

    # Read the metadata file with configured separator and no header
    try:
        metadata_df = pd.read_csv(
            metadata_path,
            sep=metadata_config["separator"],
            header=None,
            on_bad_lines='skip',
            names=['filename', 'text', 'text_norm']  # Assign column names for LJSpeech
        )
    except Exception as e:
        print(f"Failed to parse metadata.csv. Error: {e}")
        return

    # Build full paths for audio files, assuming they are in a 'wavs' subdirectory
    metadata_df['filename'] = metadata_df['filename'].apply(
        lambda x: os.path.join(local_dataset_path, 'wavs', f'{x}.wav')
    )

    # Convert the pandas DataFrame to a Hugging Face Dataset object
    dataset = Dataset.from_pandas(metadata_df)

    print("LJSpeech dataset loaded successfully.")

    # Cast the audio column to the Audio feature type with configured sampling rate
    target_sr = config.data["audio"]["target_sampling_rate"]
    print(f"Casting audio column to {target_sr}Hz sampling rate...")
    dataset = dataset.cast_column('filename', Audio(sampling_rate=target_sr))

    # Rename for consistency with the training script
    dataset = dataset.rename_column('filename', "audio")
    
    # Remove the text_norm column as it's not needed for training
    if 'text_norm' in dataset.column_names:
        dataset = dataset.remove_columns(['text_norm'])

    # Save the processed dataset in the efficient Arrow format
    print(f"Saving processed dataset to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    dataset.save_to_disk(output_dir)
    print(f"Preprocessing complete. Processed data saved to '{output_dir}'.")
    print("The dataset is now ready for training.")


def main():
    """Main function to handle command-line arguments and run preprocessing."""
    parser = argparse.ArgumentParser(
        description="Preprocess audio dataset for Spark-TTS training."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the configuration YAML file (default: config.yaml)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-processing even if output already exists"
    )

    args = parser.parse_args()

    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)

    output_data_dir = config.data["preprocessed_dataset_path"]
    dataset_name = config.data.get("dataset_name", "default")  # Get dataset name from config

    # Check if already processed
    if not args.force and os.path.exists(output_data_dir):
        print(f"Processed dataset already exists at '{output_data_dir}'.")
        print("Use --force flag to re-process the data, or delete the folder manually.")
        return

    # If force flag is set and directory exists, remove it
    if args.force and os.path.exists(output_data_dir):
        import shutil
        print(f"Removing existing processed dataset at '{output_data_dir}'...")
        shutil.rmtree(output_data_dir)

    # Run preprocessing based on dataset name
    if dataset_name.lower() == "ljspeech":
        print("Preprocessing LJSpeech dataset...")
        preprocess_ljspeech(config)
    else:
        print("Preprocessing default dataset...")
        preprocess_data_from_local(config)


if __name__ == "__main__":
    main()

