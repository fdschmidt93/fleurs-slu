import re
import string
import unicodedata
from pathlib import Path
from typing import Any


def remove_punctuation(text: str) -> str:
    """Language-agnostic of punctuation removal via unicode categories."""
    # Create a set of ASCII punctuation marks
    ascii_punctuation = set(string.punctuation)

    # Use a list comprehension to filter out characters that are classified as punctuation
    cleaned_text = "".join(
        char
        for char in text
        if char not in ascii_punctuation
        and not unicodedata.category(char).startswith("P")
    )
    return cleaned_text


def remove_extra_whitespace(sentence: str) -> str:
    """
    Normalize a given sentence by stripping leading/trailing whitespace and removing extra spaces.

    Args:
        sentence (str): The sentence to normalize.

    Returns:
        str: The normalized sentence.
    """
    return re.sub(r"\s+", " ", sentence.strip())


def read_fleurs(path: Path) -> list[dict[str, Any]]:
    """Read Fleurs tsv files into records that can easily be converted to pandas or HF datasets."""
    with open(path, "r") as file:
        lines = file.readlines()
    data = []
    for line in lines:
        (
            _id,
            file_name,
            raw_transcription,
            transcription,
            _,
            num_samples,
            speaker_id,
            gender,
        ) = line.strip().split("\t")

        # speaker_id sometimes mixes string and digit
        if speaker_id.isdigit():
            speaker_id = int(speaker_id)
        elif any(c.isdigit() for c in speaker_id):
            speaker_id = int("".join([c for c in speaker_id if c.isdigit()]))
        else:
            speaker_id = -1

        data.append(
            {
                "filename": file_name,
                "fleurs_id": int(_id),
                "raw_transcription": raw_transcription,
                "transcription": transcription,
                "num_samples": int(num_samples),
                "speaker_id": speaker_id,
                "gender": gender,
                "split": path.stem,
            }
        )
    return data


def find_project_root(
    path: str | Path,
    markers=(
        "setup.py",
        "pyproject.toml",
        "requirements.txt",
        "environment.yaml",
        "environment.yml",
        ".git",
    ),
) -> Path:
    path_ = path if isinstance(path, Path) else Path(path)
    """Recursively searches for the project root by looking for common marker files."""
    for parent in path_.resolve().parents:
        if any((parent / marker_file).exists() for marker_file in markers):
            return parent
    raise RuntimeError(f"Project root not found from {path}. No marker files found.")
