# src/preprocess_data.py

import os
import pandas as pd
from datasets import Dataset, Audio

def preprocess_data_from_local(local_dataset_path):
    """
    Loads and preprocesses a local dataset with a specific metadata format.
    - Expects metadata format: audio_file|text|speaker_name
    - The 'audio_file' column must contain the full path to the wav file.
    - Saves the processed dataset to 'data/preprocessed_dataset' for faster loading.

    Args:
        local_dataset_path (str): The path to the directory containing 'metadata.csv'.
    """
    output_dir = "data/preprocessed_dataset"
    metadata_path = os.path.join(local_dataset_path, 'metadata.csv')

    # Check if metadata file exists
    if not os.path.exists(metadata_path):
        print(f"Error: metadata.csv not found in '{local_dataset_path}'")
        return

    print(f"Loading metadata from: {metadata_path}")
    
    # --- THIS IS THE KEY CHANGE ---
    # Read the metadata file, which now has a header and full paths.
    try:
        # Use header=0 to recognize the first line as column names
        metadata_df = pd.read_csv(metadata_path, sep='|', header=0, on_bad_lines='skip')
        
        # Verify the expected columns are present
        expected_columns = ['audio_file', 'text', 'speaker_name']
        if not all(col in metadata_df.columns for col in expected_columns):
            print(f"Error: metadata.csv is missing one of the expected columns: {expected_columns}")
            print(f"Found columns: {metadata_df.columns.tolist()}")
            return
            
    except Exception as e:
        print(f"Failed to parse metadata.csv. Error: {e}")
        return

    metadata_df["audio_file"]  = local_dataset_path + "/" + metadata_df["audio_file"]

    # No need to build the path, as it's already provided.
    # Convert the pandas DataFrame to a Hugging Face Dataset object
    dataset = Dataset.from_pandas(metadata_df)
    
    print("Dataset loaded successfully from local metadata file.")
    
    # Cast the 'audio_file' column to the Audio feature type.
    # The library will use the full path in this column to load the audio.
    print("Casting audio column to 16kHz sampling rate...")
    dataset = dataset.cast_column("audio_file", Audio(sampling_rate=16000))
    
    # Rename for consistency with the training script
    dataset = dataset.rename_column("audio_file", "audio")
    # Remove the speaker_name column
    dataset = dataset.remove_columns(['speaker_name'])

    # Save the processed dataset in the efficient Arrow format
    print(f"Saving processed dataset to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    dataset.save_to_disk(output_dir)
    print(f"Preprocessing complete. Processed data saved to '{output_dir}'.")
    print("The dataset is now ready for training.")

if __name__ == "__main__":
    # --- IMPORTANT ---
    # Set the path to the directory containing your 'metadata.csv'
    # In your case, this is the main project directory.
    path_to_your_dataset_folder = "./LJSpeech-1.1/"

    output_data_dir = "data/preprocessed_dataset"
    if not os.path.exists(output_data_dir):
        preprocess_data_from_local(path_to_your_dataset_folder)
    else:
        print(f"Processed dataset already exists at '{output_data_dir}'. Skipping preprocessing.")
        print("Delete this folder if you want to re-process the data.")
