# synthesize.py

import os
import sys
import re
import torch
import argparse
import soundfile as sf
from tqdm import tqdm
from unsloth import FastModel

from config_loader import load_config

# Add Spark-TTS to path
if not os.path.exists("Spark-TTS"):
    print("Cloning Spark-TTS repository...")
    os.system("git clone https://github.com/SparkAudio/Spark-TTS")
sys.path.append('Spark-TTS')

from sparktts.models.audio_tokenizer import BiCodecTokenizer


@torch.inference_mode()
def synthesize_batch(
    model,
    tokenizer,
    audio_tokenizer,
    texts: list[str],
    output_paths: list[str],
    config,
):
    """
    Generates audio for a batch of text strings using pre-loaded models.

    Args:
        model: The pre-loaded language model.
        tokenizer: The pre-loaded text tokenizer.
        audio_tokenizer: The pre-loaded audio tokenizer.
        texts (list[str]): A batch of texts to be synthesized.
        output_paths (list[str]): A batch of paths to save the generated .wav files.
        config: Configuration object containing generation parameters.
    """
    device = model.device
    gen_config = config.synthesis["generation"]
    token_format = config.audio_tokenizer["token_format"]

    # Prepare Prompts for the entire batch
    prompt_template = (
        f"{token_format['task_token']}"
        f"{token_format['start_content']}{{text}}{token_format['end_content']}"
        f"{token_format['start_global_token']}"
    )
    prompts = [prompt_template.format(text=text) for text in texts]

    # Tokenize the whole batch at once. Padding is crucial for batching.
    tokenizer.padding_side = "left"  # For generation
    model_inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)

    # Generate Audio Tokens for the entire batch
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=gen_config["max_new_tokens"],
        do_sample=gen_config["do_sample"],
        temperature=gen_config["temperature"],
        top_k=gen_config["top_k"],
        top_p=gen_config["top_p"],
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    # Decode all generated sequences at once
    generated_ids_trimmed = generated_ids[:, model_inputs.input_ids.shape[1]:]
    predicts_texts = tokenizer.batch_decode(generated_ids_trimmed, skip_special_tokens=False)

    # Process each result in the batch
    for i, predicts_text in enumerate(predicts_texts):
        # Extract Semantic and Global Tokens
        # Escape the pipe characters in the token templates before creating regex
        semantic_template_escaped = token_format["semantic_token_template"].replace("|", r"\|")
        global_template_escaped = token_format["global_token_template"].replace("|", r"\|")
        semantic_pattern = semantic_template_escaped.replace("{id}", r"(\d+)")
        global_pattern = global_template_escaped.replace("{id}", r"(\d+)")

        semantic_matches = re.findall(semantic_pattern, predicts_text)
        global_matches = re.findall(global_pattern, predicts_text)

        if not semantic_matches or not global_matches:
            print(f"Warning: Could not find tokens for sentence {i+1}. Skipping.")
            continue

        pred_semantic_ids = torch.tensor(
            [int(token) for token in semantic_matches], dtype=torch.long, device=device
        ).unsqueeze(0)
        pred_global_ids = torch.tensor(
            [int(token) for token in global_matches], dtype=torch.long, device=device
        ).unsqueeze(0)
        pred_global_ids = pred_global_ids.unsqueeze(0)

        # Detokenize to Audio
        wav_np = audio_tokenizer.detokenize(
            pred_global_ids.squeeze(0),
            pred_semantic_ids
        )

        # Save the Audio File
        if wav_np.size > 0:
            sample_rate = config.synthesis["sample_rate"]
            sf.write(output_paths[i], wav_np, sample_rate)
        else:
            print(f"Warning: Audio generation failed for sentence {i+1}.")


def load_models(config):
    """Load all required models for synthesis."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading models onto device: {device}")

    model_path = config.synthesis["model_path"]
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model directory not found at '{model_path}'")

    # Parse dtype
    dtype_str = config.synthesis["dtype"]
    dtype_map = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16
    }
    dtype = dtype_map.get(dtype_str, torch.float32)

    # Load LLM
    model, tokenizer = FastModel.from_pretrained(
        model_name=model_path,
        max_seq_length=config.synthesis["max_seq_length"],
        dtype=dtype,
        load_in_4bit=config.synthesis["load_in_4bit"],
    )
    model.to(device)
    FastModel.for_inference(model)

    # Load audio tokenizer
    audio_tokenizer_path = config.synthesis["audio_tokenizer_path"]
    if not os.path.exists(audio_tokenizer_path):
        raise FileNotFoundError(f"Base audio tokenizer not found at '{audio_tokenizer_path}'")

    audio_tokenizer = BiCodecTokenizer(audio_tokenizer_path, device)
    print("✅ Models loaded successfully.")

    return model, tokenizer, audio_tokenizer


def read_input_file(file_path):
    """Read sentences from input file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input text file not found at '{file_path}'")

    with open(file_path, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f if line.strip()]

    return sentences


def synthesize_from_file(config, input_file=None):
    """
    Main synthesis function.

    Args:
        config: Configuration object (already loaded).
        input_file: Optional override for input file path.
    """

    # Override input file if provided
    if input_file:
        config.synthesis["input_file"] = input_file

    # Validate input file
    if not config.synthesis.get("input_file"):
        raise ValueError("Input file must be specified in config or via --input_file argument")

    # Load models
    model, tokenizer, audio_tokenizer = load_models(config)

    # Read input file
    sentences = read_input_file(config.synthesis["input_file"])
    print(f"Found {len(sentences)} sentences to synthesize.")

    # Create output directory
    output_dir = config.synthesis["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # Process sentences in batches
    batch_size = config.synthesis["batch_size"]
    skip_existing = config.synthesis["skip_existing"]

    with tqdm(total=len(sentences), desc="Synthesizing sentences") as pbar:
        for i in range(0, len(sentences), batch_size):
            # Original batch
            original_batch_texts = sentences[i:i + batch_size]
            original_batch_paths = [
                os.path.join(output_dir, f"output_{i+j+1:05d}.{config.synthesis['output_format']}")
                for j in range(len(original_batch_texts))
            ]

            # Filter batch to skip existing files if configured
            texts_to_process = []
            paths_to_process = []

            for text, path in zip(original_batch_texts, original_batch_paths):
                if not skip_existing or not os.path.exists(path):
                    texts_to_process.append(text)
                    paths_to_process.append(path)

            # Synthesize if there are texts to process
            if texts_to_process:
                print(f"\nProcessing batch. Synthesizing {len(texts_to_process)} of {len(original_batch_texts)} sentences.")
                try:
                    synthesize_batch(
                        model=model,
                        tokenizer=tokenizer,
                        audio_tokenizer=audio_tokenizer,
                        texts=texts_to_process,
                        output_paths=paths_to_process,
                        config=config
                    )
                except Exception as e:
                    print(f"\nError during batch {i//batch_size + 1}: {e}")
                    print("Skipping this filtered batch.")

            # Update progress bar
            pbar.update(len(original_batch_texts))

    print(f"\n✅ Batch synthesis complete. Output saved to: {output_dir}")


def main():
    """Main function to handle command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate speech from a text file in batches using a fine-tuned Spark-TTS model."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the configuration YAML file (default: config.yaml)"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        help="Path to a text file with one sentence per line (overrides config)"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the fine-tuned model directory (overrides config)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to save the generated .wav files (overrides config)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Number of sentences to process in a single batch (overrides config)"
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override config with command-line arguments if provided
    if args.model_path:
        config.synthesis["model_path"] = args.model_path
    if args.output_dir:
        config.synthesis["output_dir"] = args.output_dir
    if args.batch_size:
        config.synthesis["batch_size"] = args.batch_size

    # Run synthesis
    synthesize_from_file(config, args.input_file)


if __name__ == "__main__":
    main()
