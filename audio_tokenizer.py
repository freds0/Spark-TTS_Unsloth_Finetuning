# audio_tokenizer.py

import os
import sys
import torch
import torchaudio.transforms as T
from config_loader import load_config

# Add Spark-TTS to path
sys.path.append('Spark-TTS')
from sparktts.models.audio_tokenizer import BiCodecTokenizer
from sparktts.utils.audio import audio_volume_normalize

# Global variables to be initialized
audio_tokenizer = None
config = None


def initialize_tokenizer(config_obj=None, config_path="config.yaml"):
    """
    Initialize the audio tokenizer with configuration.

    Args:
        config_obj: Pre-loaded Config object (optional).
        config_path: Path to config file if config_obj is not provided.
    """
    global audio_tokenizer, config

    if config_obj is None:
        config = load_config(config_path)
    else:
        config = config_obj

    device = config.audio_tokenizer["device"]
    model_path = config.model["base_model_path"]

    print(f"Initializing audio tokenizer on device: {device}")
    audio_tokenizer = BiCodecTokenizer(model_path, device)
    print("Audio tokenizer initialized successfully.")


def extract_wav2vec2_features(wavs: torch.Tensor) -> torch.Tensor:
    """
    Extract wav2vec2 features using configured hidden state layers.

    Args:
        wavs: Audio tensor with shape (1, seq_len)

    Returns:
        Mixed features from configured hidden state layers
    """
    if audio_tokenizer is None:
        raise RuntimeError("Audio tokenizer not initialized. Call initialize_tokenizer() first.")

    if wavs.shape[0] != 1:
        raise ValueError(f"Expected batch size 1, but got shape {wavs.shape}")

    wav_np = wavs.squeeze(0).cpu().numpy()

    # Get sampling rate from config
    sampling_rate = config.audio_tokenizer["wav2vec2"]["sampling_rate"]

    processed = audio_tokenizer.processor(
        wav_np,
        sampling_rate=sampling_rate,
        return_tensors="pt",
        padding=config.audio_tokenizer["wav2vec2"]["padding"],
    )
    input_values = processed.input_values
    input_values = input_values.to(audio_tokenizer.feature_extractor.device)

    model_output = audio_tokenizer.feature_extractor(input_values)

    if model_output.hidden_states is None:
        raise ValueError("Wav2Vec2Model did not return hidden states. Ensure config `output_hidden_states=True`.")

    num_layers = len(model_output.hidden_states)
    required_layers = config.audio_tokenizer["wav2vec2"]["hidden_state_layers"]

    if any(l >= num_layers for l in required_layers):
        raise IndexError(
            f"Requested hidden state indices {required_layers} out of range for model with {num_layers} layers."
        )

    # Mix the configured hidden state layers
    feats_mix = sum(model_output.hidden_states[l] for l in required_layers) / len(required_layers)

    return feats_mix


def formatting_audio_func(example):
    """
    Format audio example for training with tokens.

    Args:
        example: Dataset example containing audio and text

    Returns:
        Dictionary with formatted text containing audio tokens
    """
    if audio_tokenizer is None:
        raise RuntimeError("Audio tokenizer not initialized. Call initialize_tokenizer() first.")

    # Get token format from config
    token_format = config.audio_tokenizer["token_format"]
    include_source = config.audio_tokenizer["include_source"]

    # Build text with optional source
    if include_source and "source" in example:
        text = f"{example['source']}: {example['text']}"
    else:
        text = example["text"]

    audio_array = example["audio"]["array"]
    sampling_rate = example["audio"]["sampling_rate"]

    target_sr = audio_tokenizer.config['sample_rate']

    # Resample if necessary
    if sampling_rate != target_sr:
        resampler = T.Resample(orig_freq=sampling_rate, new_freq=target_sr)
        audio_tensor_temp = torch.from_numpy(audio_array).float()
        audio_array = resampler(audio_tensor_temp).numpy()

    # Normalize volume if configured
    if audio_tokenizer.config["volume_normalize"]:
        audio_array = audio_volume_normalize(audio_array)

    ref_wav_np = audio_tokenizer.get_ref_clip(audio_array)

    audio_tensor = torch.from_numpy(audio_array).unsqueeze(0).float().to(audio_tokenizer.device)
    ref_wav_tensor = torch.from_numpy(ref_wav_np).unsqueeze(0).float().to(audio_tokenizer.device)

    # Extract features
    feat = extract_wav2vec2_features(audio_tensor)

    batch = {
        "wav": audio_tensor,
        "ref_wav": ref_wav_tensor,
        "feat": feat.to(audio_tokenizer.device),
    }

    # Tokenize audio
    semantic_token_ids, global_token_ids = audio_tokenizer.model.tokenize(batch)

    # Format tokens using configured templates
    global_token_template = token_format["global_token_template"]
    semantic_token_template = token_format["semantic_token_template"]

    global_tokens = "".join(
        [global_token_template.format(id=i) for i in global_token_ids.squeeze().cpu().numpy()]
    )
    semantic_tokens = "".join(
        [semantic_token_template.format(id=i) for i in semantic_token_ids.squeeze().cpu().numpy()]
    )

    # Build final input string using configured tokens
    inputs = [
        token_format["task_token"],
        token_format["start_content"],
        text,
        token_format["end_content"],
        token_format["start_global_token"],
        global_tokens,
        token_format["end_global_token"],
        token_format["start_semantic_token"],
        semantic_tokens,
        token_format["end_semantic_token"],
        token_format["end_sequence"]
    ]
    inputs = "".join(inputs)

    return {"text": inputs}


# Initialize tokenizer when module is imported (for backwards compatibility)
# This can be commented out if explicit initialization is preferred
try:
    if os.path.exists("config.yaml"):
        initialize_tokenizer()
except Exception as e:
    print(f"Warning: Could not auto-initialize tokenizer: {e}")
    print("Call initialize_tokenizer() manually before using formatting_audio_func()")
