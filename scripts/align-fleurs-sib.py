"""
Summary:

This script merges datasets from FLEURS into the splits of SIB-200 (https://huggingface.co/datasets/Davlan/sib200).
It utilizes a Levenshtein distance-based matching strategy to handle minor textual discrepancies.
The resulting datasets are uploaded to the Hugging Face Hub for further use.

Key Components:
- Logging mechanism to track processed languages and errors.
- Fuzzy matching to align sentences with a Levenshtein threshold.
- Dataset casting and transformation to conform to required formats.
- Uploading processed datasets to the Hugging Face Hub.

Usage:
Run the script directly to process and merge the datasets.
"""

import sys
from pathlib import Path

# try-except for interactive repl use
try:
    sys.path.append(str(Path(__file__).parent.parent.absolute()))
except:
    pass

from typing import cast

import pandas as pd
from datasets import Audio, ClassLabel, Dataset, Value, Sequence, load_dataset
from src.language_mappings import FLEURS_TO_FLORES
from src.levenshtein import match_sentences
from src.utils import find_project_root, remove_extra_whitespace, remove_punctuation


# Constants
LEVENSHTEIN_THRESHOLD = 3
NUM_WORKERS = 4

# Setup paths
try:
    PROJECT = find_project_root(__file__)
except NameError:  # Handle interactive use
    PROJECT = find_project_root(str(Path("./dummy_file.py").absolute()))

DATA_DIR = PROJECT / "data"
FLEURS_ASR_DIR = DATA_DIR / "flores-fleurs_asr"
SIB_DIR = DATA_DIR / "fleurs-sib"
SIB_DIR.mkdir(exist_ok=True)

LOG_FILE = PROJECT / "logs" / "fleurs-sib.txt"
LOG_ERROR = PROJECT / "logs" / "fleurs-sib-errors.txt"

# Read covered languages and splits from log file
covered_lang2split = {}
if LOG_FILE.exists():
    with open(LOG_FILE, "r") as file:
        for line in file:
            lang, split = line.strip().split("\t")[:2]
            if lang not in covered_lang2split:
                covered_lang2split[lang] = set()
            covered_lang2split[lang].add(split)


def write_to_log(message: str, error: bool = False):
        """
    Write a message to the appropriate log file.

    Args:
        message (str): The message to write.
        error (bool): Whether the message is an error message. Defaults to False.
    """
    log_path = LOG_ERROR if error else LOG_FILE
    with open(log_path, "a") as log:
        log.write(f"{message}\n")


def merge_flores_sib(language: str):
    """Merge FLEURS and SIB datasets for a given language."""
    flores_language = FLEURS_TO_FLORES[language]
    sib_lang_dir = SIB_DIR / flores_language
    sib_lang_dir.mkdir(exist_ok=True)

    for split in ("train", "validation", "test"):
        if language in covered_lang2split and split in covered_lang2split[language]:
            continue

        try:
            fleurs = pd.read_parquet(FLEURS_ASR_DIR / f"{language}.parquet")
        except FileNotFoundError:
            write_to_log(f"{language}-{split}: FLEURS-ASR file missing", error=True)
            continue

        try:
            sib = cast(
                pd.DataFrame,
                cast(
                    Dataset, load_dataset("Davlan/sib200", flores_language, split=split)
                ).to_pandas(),
            )
        except Exception:
            write_to_log(f"{language}-{split}: SIB dataset missing", error=True)
            continue

        merged = fleurs.merge(
            sib, left_on="sentence", right_on="text", how="right", indicator=True
        )
        unmatched = merged[merged["_merge"] == "right_only"]["text"].tolist()
        matched = merged["_merge"] == "both"
        merged = merged.loc[matched].drop(columns=["_merge"])

        if unmatched:
            normalized_unmatched = [
                remove_punctuation(remove_extra_whitespace(line)) for line in unmatched
            ]
            unmatched_map = dict(zip(normalized_unmatched, unmatched))

            normalized_fleurs = [
                remove_punctuation(remove_extra_whitespace(sentence))
                for sentence in fleurs["sentence"]
            ]
            fleurs_map = dict(zip(normalized_fleurs, fleurs["sentence"]))

            matches = match_sentences(
                normalized_unmatched, normalized_fleurs, LEVENSHTEIN_THRESHOLD
            )
            unmatched_to_matched = {
                fleurs_map[matches[k]]: unmatched_map[k] for k in matches
            }

            fleurs["sentence"] = fleurs["sentence"].replace(unmatched_to_matched)
            merged = fleurs.merge(
                sib, left_on="sentence", right_on="text", how="right", indicator=True
            )
            matched = merged["_merge"] == "both"
            merged = merged.loc[matched].drop(columns=["_merge"])

        audio_path_prefix = f"./data/fleurs/{language}/audio"
        merged["audio"] = [
            [
                f"{audio_path_prefix}/{split}/{filename}"
                for split, filename in zip(row["split"], row["filename"])
            ]
            for _, row in merged.iterrows()
        ]

        dataset = Dataset.from_pandas(merged, preserve_index=False)
        for col in ["id", "has_image", "has_hyperlink", "fleurs_id"]:
            dataset = dataset.cast_column(col, Value(dtype="int32"))
        dataset = dataset.cast_column(
            "audio", Sequence(feature=Audio(sampling_rate=16000))
        )
        dataset = dataset.cast_column(
            "category",
            ClassLabel(
                names=[
                    "science/technology",
                    "travel",
                    "politics",
                    "sports",
                    "health",
                    "entertainment",
                    "geography",
                ]
            ),
        )
        dataset = dataset.remove_columns(["full_paragraph"])

        try:
            dataset.push_to_hub(
                "wuenlp/fleurs-sib",
                commit_message=f"Added {flores_language}/{split}",
                config_name=flores_language,
                split=split,
                data_dir=f"data/{flores_language}",
            )
            write_to_log(f"{language}\t{split}\t{len(dataset)}/{len(sib)}")
        except Exception as e:
            write_to_log(f"Error uploading {language}-{split}: {str(e)}", error=True)

        print(f"Processed {language}-{split}")


def main():
    # INFO: this cannot be process-pooled because Huggingface upload commit cross-fire otherwise
    for language in FLEURS_TO_FLORES.keys():
        merge_flores_sib(language)


if __name__ == "__main__":
    main()
