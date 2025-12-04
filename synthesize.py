# src/synthesize.py

import os
import torch
import re
import numpy as np
import soundfile as sf
import sys
import argparse
from tqdm import tqdm
from unsloth import FastModel

# Add the Spark-TTS directory to the path to allow imports
if not os.path.exists("Spark-TTS"):
    print("Cloning Spark-TTS repository...")
    os.system("git clone https://github.com/SparkAudio/Spark-TTS")
sys.path.append('Spark-TTS')

# It's important to import after appending the path
from sparktts.models.audio_tokenizer import BiCodecTokenizer

@torch.inference_mode()
def synthesize_batch(
    model,
    tokenizer,
    audio_tokenizer,
    texts: list[str],
    output_paths: list[str],
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 1.0,
):
    """
    Generates audio for a batch of text strings using pre-loaded models.

    Args:
        model: The pre-loaded language model.
        tokenizer: The pre-loaded text tokenizer.
        audio_tokenizer: The pre-loaded audio tokenizer.
        texts (list[str]): A batch of texts to be synthesized.
        output_paths (list[str]): A batch of paths to save the generated .wav files.
        temperature (float): Sampling temperature for generation.
        top_k (int): Top-k sampling parameter.
        top_p (float): Top-p (nucleus) sampling parameter.
    """
    device = model.device

    # --- 1. Prepare Prompts for the entire batch ---
    prompts = [f"<|task_tts|><|start_content|>{text}<|end_content|><|start_global_token|>" for text in texts]
    
    # Tokenize the whole batch at once. Padding is crucial for batching.
    tokenizer.padding_side = "left" # For generation
    model_inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)

    # --- 2. Generate Audio Tokens for the entire batch ---
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=2048,
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    # Decode all generated sequences at once
    generated_ids_trimmed = generated_ids[:, model_inputs.input_ids.shape[1]:]
    predicts_texts = tokenizer.batch_decode(generated_ids_trimmed, skip_special_tokens=False)

    # --- 3. Process each result in the batch ---
    for i, predicts_text in enumerate(predicts_texts):
        # --- Extract Semantic and Global Tokens ---
        semantic_matches = re.findall(r"<\|bicodec_semantic_(\d+)\|>", predicts_text)
        global_matches = re.findall(r"<\|bicodec_global_(\d+)\|>", predicts_text)

        if not semantic_matches or not global_matches:
            print(f"Warning: Could not find tokens for sentence {i+1}. Skipping.")
            continue

        pred_semantic_ids = torch.tensor([int(token) for token in semantic_matches], dtype=torch.long, device=device).unsqueeze(0)
        pred_global_ids = torch.tensor([int(token) for token in global_matches], dtype=torch.long, device=device).unsqueeze(0)
        pred_global_ids = pred_global_ids.unsqueeze(0)

        # --- Detokenize to Audio ---
        wav_np = audio_tokenizer.detokenize(
            pred_global_ids.squeeze(0),
            pred_semantic_ids
        )

        # --- Save the Audio File ---
        if wav_np.size > 0:
            sample_rate = audio_tokenizer.config.get("sample_rate", 16000)
            sf.write(output_paths[i], wav_np, sample_rate)
        else:
            print(f"Warning: Audio generation failed for sentence {i+1}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate speech from a text file in batches using a fine-tuned Spark-TTS model.")
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to a text file with one sentence per line."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/checkpoint-1500",
        help="Path to the directory containing the fine-tuned model."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/synthesis_batch",
        help="Directory to save the generated .wav files."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Number of sentences to process in a single batch."
    )
    
    args = parser.parse_args()

    # --- Load models ONCE ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading models onto device: {device}")

    if not os.path.exists(args.model_path):
        print(f"Error: Model directory not found at '{args.model_path}'")
        sys.exit(1)

    model, tokenizer = FastModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=128,
        dtype = torch.float32, 
        load_in_4bit=False,
    )
    model.to(device)
    FastModel.for_inference(model)

    audio_tokenizer_path = "models/Spark-TTS-0.5B"
    if not os.path.exists(audio_tokenizer_path):
         print(f"Error: Base audio tokenizer not found at '{audio_tokenizer_path}'")
         sys.exit(1)
         
    audio_tokenizer = BiCodecTokenizer(audio_tokenizer_path, device)
    print("✅ Models loaded successfully.")

    # --- Read input file ---
    if not os.path.exists(args.input_file):
        print(f"Error: Input text file not found at '{args.input_file}'")
        sys.exit(1)
        
    with open(args.input_file, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f if line.strip()]

    os.makedirs(args.output_dir, exist_ok=True)
    
    # --- Process sentences in mini-batches ---
    total_batches = (len(sentences) + args.batch_size - 1) // args.batch_size
    
    # --- Process sentences in mini-batches ---
    with tqdm(total=len(sentences), desc="Synthesizing sentences") as pbar:
        for i in range(0, len(sentences), args.batch_size):
            # Cria o lote original de textos e seus respectivos caminhos de saída
            original_batch_texts = sentences[i:i + args.batch_size]
            original_batch_paths = [os.path.join(args.output_dir, f"output_{i+j+1:05d}.wav") for j in range(len(original_batch_texts))]
            
            # Filtra o lote, mantendo apenas os itens cujo arquivo de saída não existe
            texts_to_process = []
            paths_to_process = []
            for text, path in zip(original_batch_texts, original_batch_paths):
                if not os.path.exists(path):
                    texts_to_process.append(text)
                    paths_to_process.append(path)
            
            # Executa a síntese somente se houver textos a serem processados no lote
            if texts_to_process:
                print(f"\nProcessando lote. Sintetizando {len(texts_to_process)} de {len(original_batch_texts)} sentenças.")
                try:
                    synthesize_batch(
                        model=model,
                        tokenizer=tokenizer,
                        audio_tokenizer=audio_tokenizer,
                        texts=texts_to_process,
                        output_paths=paths_to_process
                    )
                except Exception as e:
                    print(f"\nOcorreu um erro durante o lote {i//args.batch_size + 1}: {e}")
                    print("Pulando este lote filtrado.")
            
            # Atualiza a barra de progresso com o tamanho do lote original para refletir o avanço total
            pbar.update(len(original_batch_texts))
    
    print("\nBatch synthesis complete.")