"""
This Python script processes audio datasets in the FLEURS format to identify and exclude audio files
that primarily contain silence. It uses a voice activity detection (VAD) model to analyze audio files
and generates cleaned datasets without silent audio files.
The VAD model also implicitly captures in audible examples.

The only false-positive that was detected from manual verification of ~50 samples was
de_dk train 4182703406352481327.wav
which is quite noisy but could be understood when listened to carefully.
This script was motivated by the observations described in: https://huggingface.co/datasets/google/fleurs/discussions/16

**IMPORTANT**

The silent and noisy files are excluded when Flores and Fleurs are aligned in ./align-flores-and-fleurs.py

Key functionalities:
1. **Normalize Audio Loudness**: Ensures audio waveforms are normalized to a target RMS level for consistent processing.
2. **Silence Detection**:
    - Uses the Silero VAD model to classify audio files as "silent" or "non-silent."
    - Applies configurable thresholds to determine if an audio file is predominantly silent.
3. **Directory Management**:
    - Organizes input and output datasets.
    - Creates logs and separate directories for processed audio files.
4. **Parallel Processing**:
    - Leverages Python's `ProcessPoolExecutor` for parallel processing of multiple languages and dataset splits
      (e.g., train, dev, test).
5. **Logging**:
    - Logs summary statistics and per-file silence classifications for each language and dataset split.

Key modules used:
- `torch` and `torchaudio` for audio processing.
- `silero-vad` for voice activity detection.
- `tqdm` for progress visualization.
- `concurrent.futures` for parallel execution.

Expected directory structure:
- Input data: Stored in `PROJECT/data/fleurs`.
- Processed data: **Symlinked** audio to `PROJECT/data/fleurs_excl_silence`.
- Logs: Saved in `PROJECT/logs/fleurs-silence`.

Usage:
Run the script directly. It will process all languages defined in the `FLEURS` mapping,
analyzing train, dev, and test splits by default.
"""

import sys
from pathlib import Path

# try-except for interactive repl use
try:
    sys.path.append(str(Path(__file__).parent.parent.absolute()))
except:
    pass

import os
from src.utils import (
    find_project_root,
)
from src.language_mappings import FLEURS
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm

import torch
import torchaudio
from torchaudio.transforms import Resample

from multiprocessing import Lock

write_lock = Lock()

try:
    PROJECT = find_project_root(__file__)
except:  # type: ignore
    # for interactive use
    FILE = str(Path("./dummy_file.py").absolute())
    PROJECT = find_project_root(FILE)
LOG_DIR = PROJECT / "logs" / "fleurs-silence"
LOG_DIR.mkdir(exist_ok=True)
DATA_DIR = PROJECT / "data"
DATA_DIR.mkdir(exist_ok=True)
FLEURS_DIR = DATA_DIR / "fleurs"
FLEURS_EXCL_SILENCE_DIR = PROJECT / "data" / "fleurs_excl_silence"
FLEURS_EXCL_SILENCE_DIR.mkdir(exist_ok=True)
NUM_WORKERS = os.cpu_count()


def normalize_loudness(waveform, target_level=-25.0):
    """
    Normalize the loudness of the waveform to a target RMS level.

    Args:
        waveform (torch.Tensor): The waveform tensor (1D or 2D).
        target_level (float): Target RMS level in decibels (dB).

    Returns:
        torch.Tensor: Loudness-normalized waveform.
    """
    rms = waveform.pow(2).mean().sqrt()  # Calculate the RMS of the waveform
    rms_db = 20 * torch.log10(rms + 1e-9)  # Convert RMS to decibels
    gain_db = target_level - rms_db  # Calculate gain needed to reach target
    gain = 10 ** (gain_db / 20)  # Convert gain from dB to linear scale
    return waveform * gain


def classify_silence_with_vad(
    audio_file_path,
    vad_model,
    get_speech_timestamps,
    silence_threshold=0.95,
    sampling_rate=16000,
    target_loudness=-25.0,
):
    """
    Classifies whether an audio file is mostly silence using Silero VAD.

    Args:
        audio_file_path (str): Path to the `.wav` audio file.
        silence_threshold (float): Fraction of frames classified as silent to classify the whole file as silence.
        sampling_rate (int): Sampling rate required by the VAD model (default is 16000).

    Returns:
        bool: True if the audio file is mostly silence, False otherwise.
    """
    # Load the Silero VAD model

    # Load the audio file
    waveform, original_rate = torchaudio.load(audio_file_path)

    # Resample if needed
    if original_rate != sampling_rate:
        resample = Resample(orig_freq=original_rate, new_freq=sampling_rate)
        waveform = resample(waveform)

    # Normalize loudness
    waveform = normalize_loudness(waveform, target_level=target_loudness)

    # Apply VAD
    audio_tensor = waveform.squeeze(0)  # Ensure tensor is 1D
    speech_timestamps = get_speech_timestamps(
        audio_tensor, vad_model, sampling_rate=sampling_rate
    )

    # Calculate the proportion of non-silent audio
    total_audio_duration = len(audio_tensor) / sampling_rate
    total_speech_duration = sum(
        [(ts["end"] - ts["start"]) / sampling_rate for ts in speech_timestamps]
    )

    non_silent_ratio = total_speech_duration / total_audio_duration

    # Classify as silence if non-silent ratio is below the threshold
    return non_silent_ratio < (1 - silence_threshold)


def clean_directory(directory: Path) -> None:
    """
    Removes all symlinks in the specified directory.

    Args:
        directory (Path): Directory to clean.
    """
    for file in directory.glob("*"):
        if file.is_symlink():
            file.unlink()


def log_summary(lang: str, split: str, silent_count: int, total_count: int) -> None:
    """
    Logs the processing summary to a global log file.

    Args:
        lang (str): Language code.
        split (str): Dataset split (e.g., "train").
        silent_count (int): Number of silent files identified.
        total_count (int): Total number of files processed.
    """
    with write_lock:
        with open(LOG_DIR / "log.txt", "a") as f:
            f.write(f"{lang}\t{split}\t{silent_count}/{total_count}\n")


def write_split_log(file_path: Path, files: list[Path], silent: list[bool]) -> None:
    """
    Writes a per-split log of silence classification results.

    Args:
        file_path (Path): Path to save the log file.
        files (list[Path]): List of audio files processed.
        silent (list[bool]): Corresponding list of silence classifications.
    """
    with open(file_path, "w") as f:
        for file, is_silent in zip(files, silent):
            f.write(f"{file.name}\t{str(is_silent)}\n")


def process_lang(lang: str, splits: tuple[str, ...] = ("train", "dev", "test")) -> None:
    """
    Processes audio files for a specific language to identify and exclude silent files.
    Symlinks the files to new directory that excludes silence (and noise)!

    Args:
        lang (str): Language code (e.g., 'en', 'fr') to process.
        splits (tuple[str, ...]): Dataset splits to process (default: ("train", "dev", "test")).

    Returns:
        None
    """
    # Load VAD model and utilities
    vad_model, vad_utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True
    )
    get_speech_timestamps = vad_utils[0]

    # Directory setup
    lang_dir = FLEURS_DIR / lang / "audio"
    lang_dir_excl_silence = FLEURS_EXCL_SILENCE_DIR / lang / "audio"
    lang_log_dir = LOG_DIR / lang
    lang_log_dir.mkdir(parents=True, exist_ok=True)
    lang_dir_excl_silence.mkdir(parents=True, exist_ok=True)

    # Process each split
    for split in splits:
        lang_split_dir = lang_dir / split
        lang_split_excl_silent_dir = lang_dir_excl_silence / split
        lang_split_excl_silent_dir.mkdir(parents=True, exist_ok=True)

        # Clean and prepare directories
        clean_directory(lang_split_excl_silent_dir)

        # Load audio files
        audio_files = list(lang_split_dir.glob("*.wav"))
        is_silent_files = []

        # Process audio files
        for wav_file in tqdm(
            audio_files, desc=f"Processing {lang}-{split}", unit="file"
        ):
            is_silent_files.append(
                classify_silence_with_vad(
                    wav_file,
                    vad_model=vad_model,
                    get_speech_timestamps=get_speech_timestamps,
                )
            )

        # Create symlinks for non-silent files
        # for file, is_silent in zip(audio_files, is_silent_files):
        #     if not is_silent:
        #         new_path = lang_split_excl_silent_dir / file.name
        #         old_path = lang_split_dir / file.name
        #         assert old_path.exists()
        #         new_path.symlink_to(old_path)

        # Log results
        log_summary(lang, split, sum(is_silent_files), len(is_silent_files))
        write_split_log(lang_log_dir / f"{split}.tsv", audio_files, is_silent_files)


def main():
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        executor.map(process_lang, FLEURS)


if __name__ == "__main__":
    main()
