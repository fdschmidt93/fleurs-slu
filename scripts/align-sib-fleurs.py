"""
This script merges sentence-aligned data from SIB-200 (https://huggingface.co/datasets/Davlan/sib200) into the splits of FLEURS.
It utilizes a Levenshtein distance-based matching strategy to handle minor textual discrepancies.

The script:
1. Iterates through a set of languages defined by a mapping from FLEURS language codes to FLORES language codes.
2. For each language:
   - Loads the corresponding FLEURS-ASR dataset and the SIB dataset splits (train, validation, test).
   - Merges the two datasets on sentence text, first by direct matching and then by approximate matching using
     Levenshtein distance for unmatched sentences.
   - Once merged, it prepares the final dataset with audio references and metadata.
   - The processed datasets are pushed to the Hugging Face Hub with their respective splits (train, validation, test).
   - A log is maintained to record processed languages and any errors encountered.
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
REPOSITORY = "wuenlp/sib-fleurs"
LEVENSHTEIN_THRESHOLD = 3
NUM_WORKERS = 4

# Setup paths
try:
    PROJECT = find_project_root(__file__)
except NameError:  # Handle interactive use
    PROJECT = find_project_root(str(Path("./dummy_file.py").absolute()))

DATA_DIR = PROJECT / "data"
FLEURS_ASR_DIR = DATA_DIR / "flores-fleurs_asr"

LOG_FILE = PROJECT / "logs" / "sib-fleurs.txt"
LOG_ERROR = PROJECT / "logs" / "sib-fleurs-errors.txt"

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
    """Write a message to the log file."""
    log_path = LOG_ERROR if error else LOG_FILE
    with open(log_path, "a") as log:
        log.write(f"{message}\n")


def process_and_upload(
    filtered_dataset: pd.DataFrame, split_name: str, language: str, flores_language: str
) -> None:
    """
    Push the filtered dataset to the HuggingFace Hub and record the results in a log file.

    This function checks whether the given split of the language dataset has already been processed.
    If not, it converts the DataFrame to a Hugging Face Dataset, adds necessary type information,
    then pushes it to the Hugging Face Hub. The action is logged in a local log file.

    :param filtered_dataset: The filtered DataFrame containing the merged and matched data.
    :param split_name: The name of the dataset split (e.g. "train", "validation", "test").
    :param language: The language code of the dataset being processed (FLEURS format).
    :param flores_language: The corresponding FLORES language code for the dataset.
    """

    if language in covered_lang2split and split_name in covered_lang2split[language]:
        print(f"{flores_language}-{split_name} already completed. Skipping.")
        return

    dataset = Dataset.from_pandas(filtered_dataset, preserve_index=False)
    for col in ["id", "has_image", "has_hyperlink", "fleurs_id"]:
        dataset = dataset.cast_column(col, Value(dtype="int32"))
    dataset = dataset.cast_column("audio", Sequence(feature=Audio(sampling_rate=16000)))
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
        dataset.remove_columns(["split"]).push_to_hub(
            REPOSITORY,
            commit_message=f"Added {flores_language}/{split_name}",
            config_name=flores_language,
            split=split_name,
            data_dir=f"data/{flores_language}",
        )
        write_to_log(f"{language}\t{split_name}\t{len(filtered_dataset)}")
    except Exception as e:
        write_to_log(f"Error uploading {language}-{split_name}: {str(e)}", error=True)


def merge_flores_sib(language: str):
    """
    Merge the SIB into FLEURS for a specified language.

    Steps:
    1. Load the language-specific FLEURS-ASR dataset.
    2. Load and concatenate the corresponding SIB dataset splits (train, validation, test).
    3. Attempt a direct merge between FLEURS-ASR and SIB by sentence text.
    4. For sentences not directly matched, apply a Levenshtein distance-based matching to find close matches.
    5. Once matched, the final merged dataset is split into train/validation/test subsets based on their original splits.
    6. Each subset is uploaded to the Hugging Face Hub, and logging is updated accordingly.

    :param language: The FLEURS language code (e.g. "en", "fr") to process.
    """
    flores_language = FLEURS_TO_FLORES[language]

    try:
        fleurs = pd.read_parquet(FLEURS_ASR_DIR / f"{language}.parquet")
    except FileNotFoundError:
        write_to_log(f"{language}: FLEURS-ASR file missing", error=True)
        return

    sibs = []
    try:
        for split in ("train", "validation", "test"):
            sibs.append(
                cast(
                    pd.DataFrame,
                    cast(
                        Dataset,
                        load_dataset("Davlan/sib200", flores_language, split=split),
                    ).to_pandas(),
                )
            )
        sib = pd.concat(sibs)
    except Exception:
        write_to_log(f"{language}: SIB dataset missing", error=True)
        return

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

    # INFO: hardcoded path here not a problem since it will get resolved upon upload
    #       HF uploads the actual audio rather than a reference
    audio_path_prefix = f"./data/fleurs/{language}/audio"
    merged["audio"] = [
        [
            f"{audio_path_prefix}/{split}/{filename}"
            for split, filename in zip(row["split"], row["filename"])
        ]
        for _, row in merged.iterrows()
    ]
    # INFO: there are 1-2 sentences that belong to multiple splits, that's why if any is train to train
    train = merged.loc[merged.split.apply(lambda example: "train" in example)]
    process_and_upload(train, "train", language, flores_language)
    validation = merged.loc[
        merged.split.apply(lambda example: all(x == "dev" for x in example))
    ]
    process_and_upload(validation, "validation", language, flores_language)
    test = merged.loc[
        merged.split.apply(lambda example: all(x == "test" for x in example))
    ]
    process_and_upload(test, "test", language, flores_language)

    print(f"Processed {language}")


def main():
    # INFO: this cannot be process-pooled because Huggingface upload commit cross-fire otherwise
    for language in FLEURS_TO_FLORES.keys():
        merge_flores_sib(language)


if __name__ == "__main__":
    main()
