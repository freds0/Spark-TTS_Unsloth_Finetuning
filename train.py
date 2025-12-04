# src/train.py

import os
import torch
import sys
from datasets import load_from_disk
from unsloth import FastModel
from trl import SFTTrainer, SFTConfig
from huggingface_hub import snapshot_download

from audio_tokenizer import formatting_audio_func, audio_tokenizer

# Add Spark-TTS to the Python path
if not os.path.exists("Spark-TTS"):
    print("Cloning Spark-TTS repository...")
    os.system("git clone https://github.com/SparkAudio/Spark-TTS")
sys.path.append('Spark-TTS')

# It's important to import after appending the path
#from sparktts.models.audio_tokenizer import BiCodecTokenizer

def train_spark_tts():
    """
    Trains the Spark-TTS model on the preprocessed dataset.
    """
    # --- 1. Model and Tokenizer Loading ---
    max_seq_length = 2048
    #model_dir = "models/Spark-TTS-0.5B"
    model_dir = "./outputs/checkpoint-1500/"
    
    if not os.path.exists(model_dir):
        print("Downloading pre-trained Spark-TTS model...")
        snapshot_download("unsloth/Spark-TTS-0.5B", local_dir=model_dir, max_workers=8)

    model, tokenizer = FastModel.from_pretrained(
        model_name = f"models/Spark-TTS-0.5B/LLM",
        #model_name = "/home/fred/Projetos/Spark-TTS-Unsloth/checkpoints_sparktts_peppa/",
        max_seq_length = max_seq_length,
        dtype = torch.float32, # Spark seems to only work on float32 for now
        full_finetuning = True, # We support full finetuning now!
        load_in_4bit = False,
        #token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )

    #LoRA does not work with float32 only works with bfloat16 !!!
    model = FastModel.get_peft_model(
        model,
        r = 128, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 128,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )

    # --- 2. Data Preparation ---
    dataset_path = "data/preprocessed_dataset"
    if not os.path.exists(dataset_path):
        print(f"Error: Processed dataset not found at '{dataset_path}'.")
        print("Please run 'python src/preprocess_data.py' first.")
        return
        
    dataset = load_from_disk(dataset_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    #audio_tokenizer = BiCodecTokenizer(model_dir, device)

    # Apply the mapping function
    #dataset = dataset.map(formatting_audio_func, remove_columns=["audio", "speaker_name"])
    
    dataset = dataset.map(formatting_audio_func, remove_columns=["audio"])
    print("Moving Bicodec model and Wav2Vec2Model to cpu.")
    audio_tokenizer.model.cpu()
    audio_tokenizer.feature_extractor.cpu()
    torch.cuda.empty_cache()

    # Free up GPU memory
    audio_tokenizer.model.cpu()
    torch.cuda.empty_cache()

    # --- 3. Training ---
    model = FastModel.get_peft_model(
        model,
        r=16, # LoRA rank
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=3407,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        packing=False,
        args=SFTConfig(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=10,
            #num_train_epochs = 5, # Set this for 1 full training run.
            max_steps = 30,            
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
            report_to = "none", # Use this for WandB etc
        ),
    )

    print("✅ Setup complete. Starting training...")
    trainer.train()
    print("🏁 Training finished.")

    # --- 4. Save the final model ---
    final_model_path = "output/final_checkpoint"
    print(f"Saving the fine-tuned model to {final_model_path}...")
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print("Model saved successfully.")

if __name__ == "__main__":
    train_spark_tts()
