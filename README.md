# Spark-TTS Fine-tuning on LJSpeech

This repository contains scripts to fine-tune the Spark-TTS model on the LJSpeech dataset for text-to-speech (TTS) synthesis.

## Project Structure

```
spark-tts-ljspeech/
│
├── data/
│   └── LJSpeech-1.1/
│
├── models/
│   └── spark-tts-ljspeech/
│
├── outputs/
│   └── generated_speech.wav
│
├── src/
│   ├── preprocess_data.py
│   ├── train.py
│   └── synthesize.py
│
├── requirements.txt
└── README.md
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd spark-tts-ljspeech
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    
    *Note: Ensure you have a compatible PyTorch version with CUDA support for GPU acceleration.*

## Usage

### 1. Data Preprocessing

First, you need to download and preprocess the LJSpeech dataset.

```bash
python src/preprocess_data.py
```

This script will download the dataset from Hugging Face Hub and save the processed version in the `data/LJSpeech-1.1-processed` directory.

### 2. Training

To fine-tune the Spark-TTS model, run the training script:

```bash
python src/train.py
```

This will:
- Download the pre-trained `unsloth/Spark-TTS-0.5B` model.
- Load the preprocessed LJSpeech dataset.
- Fine-tune the model and save the result to `models/spark-tts-ljspeech`.

### 3. Inference (Text-to-Speech)

After training, you can generate speech from text using the `synthesize.py` script.

```bash
python src/synthesize.py
```

The script will load your fine-tuned model and generate an audio file `outputs/generated_speech.wav` from the sample text inside the script. You can modify the `text_to_synthesize` variable in `synthesize.py` to generate different sentences.