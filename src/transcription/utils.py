import json
from pathlib import Path

import torch
import torchaudio
from mutagen.wave import WAVE
from typing import Any


def get_wav_metadata_duration(filepath: Path) -> float:
    """
    Retrieve the duration of a WAV audio file.

    Args:
        filepath (Path): The path to the WAV file.

    Returns:
        float: Duration of the audio in seconds.
    """
    audio = WAVE(filepath)
    return audio.info.length  # Duration in seconds


def write_to_ndjson(data: list[dict[str, Any]], output_file: Path):
    """
    Appends data to a JSON Lines (NDJSON) file.
    Each entry in `data` is a dictionary that will be written as a single line in the file.
    """
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_audio(file_path: Path, target_sample_rate: int = 16000) -> torch.Tensor:
    """
    Load an audio file and resample it to the target sample rate if necessary.

    Args:
        file_path (Path): The path to the audio file.
        target_sample_rate (int, optional): The desired sample rate for the audio file. Defaults to 16000.

    Returns:
        torch.Tensor: The loaded audio waveform as a tensor.
    """
    waveform, sample_rate = torchaudio.load(file_path)
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
        waveform = resampler(waveform)
    return waveform.squeeze()
