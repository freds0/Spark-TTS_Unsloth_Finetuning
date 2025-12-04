#@title Tokenization Function

import locale
import torchaudio.transforms as T
import os
import torch
import sys
import numpy as np
sys.path.append('Spark-TTS')
from sparktts.models.audio_tokenizer import BiCodecTokenizer
from sparktts.utils.audio import audio_volume_normalize

audio_tokenizer = BiCodecTokenizer("models/Spark-TTS-0.5B", "cuda")


def extract_wav2vec2_features( wavs: torch.Tensor) -> torch.Tensor:
        """extract wav2vec2 features"""

        if wavs.shape[0] != 1:

             raise ValueError(f"Expected batch size 1, but got shape {wavs.shape}")
        wav_np = wavs.squeeze(0).cpu().numpy()

        processed = audio_tokenizer.processor(
            wav_np,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        input_values = processed.input_values

        input_values = input_values.to(audio_tokenizer.feature_extractor.device)

        model_output = audio_tokenizer.feature_extractor(
            input_values,
        )


        if model_output.hidden_states is None:
             raise ValueError("Wav2Vec2Model did not return hidden states. Ensure config `output_hidden_states=True`.")

        num_layers = len(model_output.hidden_states)
        required_layers = [11, 14, 16]
        if any(l >= num_layers for l in required_layers):
             raise IndexError(f"Requested hidden state indices {required_layers} out of range for model with {num_layers} layers.")

        feats_mix = (
            model_output.hidden_states[11] + model_output.hidden_states[14] + model_output.hidden_states[16]
        ) / 3

        return feats_mix
def formatting_audio_func(example):
    text = f"{example['source']}: {example['text']}" if "source" in example else example["text"]
    audio_array = example["audio"]["array"]
    sampling_rate = example["audio"]["sampling_rate"]

    target_sr = audio_tokenizer.config['sample_rate']

    if sampling_rate != target_sr:
        resampler = T.Resample(orig_freq=sampling_rate, new_freq=target_sr)
        audio_tensor_temp = torch.from_numpy(audio_array).float()
        audio_array = resampler(audio_tensor_temp).numpy()

    if audio_tokenizer.config["volume_normalize"]:
        audio_array = audio_volume_normalize(audio_array)

    ref_wav_np = audio_tokenizer.get_ref_clip(audio_array)

    audio_tensor = torch.from_numpy(audio_array).unsqueeze(0).float().to(audio_tokenizer.device)
    ref_wav_tensor = torch.from_numpy(ref_wav_np).unsqueeze(0).float().to(audio_tokenizer.device)


    feat = extract_wav2vec2_features(audio_tensor)

    batch = {

        "wav": audio_tensor,
        "ref_wav": ref_wav_tensor,
        "feat": feat.to(audio_tokenizer.device),
    }


    semantic_token_ids, global_token_ids = audio_tokenizer.model.tokenize(batch)

    global_tokens = "".join(
        [f"<|bicodec_global_{i}|>" for i in global_token_ids.squeeze().cpu().numpy()] # Squeeze batch dim
    )
    semantic_tokens = "".join(
        [f"<|bicodec_semantic_{i}|>" for i in semantic_token_ids.squeeze().cpu().numpy()] # Squeeze batch dim
    )

    inputs = [
        "<|task_tts|>",
        "<|start_content|>",
        text,
        "<|end_content|>",
        "<|start_global_token|>",
        global_tokens,
        "<|end_global_token|>",
        "<|start_semantic_token|>",
        semantic_tokens,
        "<|end_semantic_token|>",
        "<|im_end|>"
    ]
    inputs = "".join(inputs)
    return {"text": inputs}


