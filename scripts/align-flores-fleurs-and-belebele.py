"""
This script merges and processes multilingual data from the Fleurs, Flores, and Belebele datasets.
It aligns the Belebele test subset with the corresponding segments from the Fleurs-Flores data.
Specifically, it:
1. Reads preprocessed Fleurs-Flores data from parquet files.
2. Loads Belebele test data for a given language.
3. Matches concatenated passages in Belebele with their corresponding Fleurs-Flores segments.
4. Asserts that the associated audio files exist, then enriches the Belebele data with these audio segments.
5. Uploads the merged and aligned dataset to a Hugging Face Hub repository.

This script is part of a data processing pipeline for multilingual ASR (Automatic Speech Recognition)
and listening comprehension experiments.
"""

import sys
from pathlib import Path

# try-except for interactive repl use
try:
    sys.path.append(str(Path(__file__).parent.parent.absolute()))
except:
    pass

from src.language_mappings import FLEURS_TO_FLORES, BELEBELE
from src.utils import find_project_root
from src.belebele import merge_fleurs_into_belebele
from datasets.load import load_dataset
from datasets.arrow_dataset import Dataset
from typing import cast
import pandas as pd
import argparse
from datasets import Audio, Sequence

try:
    PROJECT = find_project_root(__file__)
except NameError:  # Handle interactive use
    PROJECT = find_project_root(str(Path("./dummy_file.py").absolute()))

REPOSITORY = "wuenlp/belebele-fleurs"
NUM_WORKERS = 8
LOG_FILE = PROJECT / "logs" / "belebele_aligned.tsv"
LOG_ERROR_DIR = PROJECT / "logs" / "belebele_aligned_error"
BELEBELE_DATA_DIR = PROJECT / "data" / "belebele_asr"


# Read covered languages and splits from log file
covered_lang2split = {}
if LOG_FILE.exists():
    with open(LOG_FILE, "r") as file:
        for line in file:
            lang, num_samples = line.strip().split("\t")
            if lang not in covered_lang2split:
                covered_lang2split[lang] = set()
            covered_lang2split[lang] = int(num_samples)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge and process Fleurs, Flores, and Belebele datasets."
    )
    parser.add_argument(
        "--language",
        type=str,
        help="Fleurs language code for the datasets. Use 'all' for all languages.",
    )
    args = parser.parse_args()
    return args


def write_stats_to_tsv(file_path: Path, language: str, retained_paragraphs: int):
    # If file doesn't exist, write headers first
    with open(file_path, "w" if not file_path.exists() else "a") as f:
        f.write(f"{language}\t{retained_paragraphs}\n")


def align(language: str) -> Dataset:
    """
    Align the Belebele test data with the corresponding Fleurs-Flores paragraphs.

    Steps:
        1. Determine the corresponding Flores language code for the given Fleurs language.
        2. Load the combined Fleurs-Flores dataset for the specified language.
        3. Filter down to full paragraphs only.
        4. Load the Belebele test data for the corresponding Flores language.
        5. Merge the Belebele dataset with the aligned Fleurs-Flores segments.
        6. Verify audio file existence and attach audio metadata.
        7. Cast the dataset with the proper features and return it.

    Args:
        language (str): The Fleurs language code.

    Returns:
        Dataset: A Hugging Face `Dataset` object containing the merged and aligned data.
    """
    flores_language = FLEURS_TO_FLORES[language]

    # Load datasets
    path = PROJECT / "data" / "flores-fleurs_asr" / f"{language}.parquet"
    fleurs_flores_df = cast(
        pd.DataFrame,
        cast(
            Dataset, load_dataset("parquet", data_files=str(path), split="train")
        ).to_pandas(),
    )
    fleurs_flores_df = fleurs_flores_df.loc[fleurs_flores_df.full_paragraph == True]
    belebele_df = cast(
        Dataset, load_dataset("facebook/belebele", flores_language, split="test")
    ).to_pandas()

    # Match concatenated passages in belebele with flores
    matched_belebele_dataset = merge_fleurs_into_belebele(belebele_df, fleurs_flores_df)

    for line in matched_belebele_dataset:
        for sent in line["sentence_data"]:
            splits = sent["split"]
            filenames = sent["filename"]
            paths = [
                f"./data/fleurs/{language}/audio/{split}/{filename}"
                for filename, split in zip(filenames, splits)
            ]
            for path in paths:
                assert Path(path).exists()
            sent["audio"] = [
                {
                    "path": f"./data/fleurs/{language}/audio/{split}/{filename}",
                    "sampling_rate": 16000,
                }
                for filename, split in zip(filenames, splits)
            ]
    dataset_ = Dataset.from_list(matched_belebele_dataset)
    feat = dataset_.features.copy()
    feat["sentence_data"][0]["audio"] = Sequence(Audio(sampling_rate=16000))
    dataset_ = dataset_.cast(feat)
    return dataset_


def mapper(language: str):
    """
    Process a single language by aligning and uploading its dataset to the Hub if not already done.

    This function:
    - Determines the Flores language code from the given Fleurs code.
    - If the language is present in Belebele and not already processed, it will:
        1. Align the dataset.
        2. Push it to the Hugging Face Hub.
        3. Write statistics to a log file.
    - If any error occurs during upload, it logs the error in the logs directory.

    Args:
        language (str): The Fleurs language code.
    """
    flores_language = FLEURS_TO_FLORES[language]
    if flores_language in BELEBELE:
        if language not in covered_lang2split:
            lang_dataset = align(language)
            try:
                lang_dataset.push_to_hub(
                    REPOSITORY,
                    config_name=flores_language,
                    split="test",
                    data_dir=f"data/{flores_language}",
                )
                print(f"{language} comprises {len(lang_dataset)} of 900 Belebele rows.")
                write_stats_to_tsv(LOG_FILE, language, len(lang_dataset))
            except Exception as e:
                # Capture the error message as a string
                # LOG_ERROR_dir is a pathlib.Path
                print(language, "error")
                print(str(e))
                with open(LOG_ERROR_DIR.joinpath(f"{language}.txt"), "w") as file:
                    file.write(str(e) + "\n")
    else:
        print(f"{language} | {flores_language} not in Belebele.")


def main(args):
    """
    Main entry point for the script.
    
    If no language is specified, it attempts to process all supported languages.
    Otherwise, it processes only the specified language.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    # INFO: this cannot be process-pooled because Huggingface upload commit cross-fire otherwise
    if args.language is None:
        for language in FLEURS_TO_FLORES.keys():
            mapper(language)
    else:
        mapper(args.language)


if __name__ == "__main__":
    args = parse_args()
    main(args)
