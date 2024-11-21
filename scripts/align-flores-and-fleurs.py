import sys
from pathlib import Path

try:
    sys.path.append(str(Path(__file__).parent.parent.absolute()))
except:
    pass

import argparse
import os
import threading
import csv
from concurrent.futures import ThreadPoolExecutor
from typing import cast

import pandas as pd
from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from src.language_mappings import FLEURS_TO_FLORES
from src.levenshtein import match_sentences
from src.utils import (
    find_project_root,
    read_fleurs,
    remove_extra_whitespace,
    remove_punctuation,
)

try:
    PROJECT = find_project_root(__file__)
except:  # type: ignore
    # for interactive use
    FILE = str(Path("./dummy_file.py").absolute())
    PROJECT = find_project_root(FILE)
LOG_FILE = PROJECT / "logs" / "flores-fleurs.csv"
DATA_DIR = PROJECT / "data" / "flores-fleurs_raw"
DATA_DIR.mkdir(exist_ok=True)
NUM_WORKERS = os.cpu_count()

# number of chars that may differ
LEVENSHTEIN_THRESHOLD = 3

write_lock = threading.Lock()


def write_stats_to_csv(
    file_path: Path,
    language: str,
    num_fleurs_sent: int,
    num_exact_matches: int,
    num_missing: int,
    num_recovered: int,
    num_total_paragraphs: int,
    num_retained_paragraphs: int,
    num_missing_paragraphs: int,
):
    """
    Write statistics to a CSV file. Creates a new file with headers if it doesn't exist,
    otherwise appends the new row to the existing file.
    """
    headers = [
        "Language",
        "FLEURS Sentences",
        "Exact Matches",
        "Difference",
        "Recovered",
        "Total Paragraphs",
        "Retained Paragraphs",
        "Missing Paragraphs",
    ]

    row = [
        language,
        num_fleurs_sent,
        num_exact_matches,
        num_missing,
        num_recovered,
        num_total_paragraphs,
        num_retained_paragraphs,
        num_missing_paragraphs,
    ]

    # If file doesn't exist, write headers first
    if not file_path.exists():
        with write_lock:
            with file_path.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                writer.writerow(row)
    else:
        # Append row to existing file
        with write_lock:
            with file_path.open("a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge and process Fleurs, Flores, and Belebele datasets."
    )
    parser.add_argument(
        "--language",
        type=str,
        help="Fleurs language code for the datasets. Do not set for all languages.",
    )
    args = parser.parse_args()
    return args


def align(language: str, log_file: Path) -> Dataset:
    FLEURS_DIR = PROJECT / "data" / "fleurs" / language
    silent_statistics_dir = PROJECT / "logs" / "fleurs-silence" / language

    flores_language = FLEURS_TO_FLORES[language]

    # Load datasets as pd.DataFrame
    flores_df = cast(
        pd.DataFrame,
        cast(
            Dataset,
            load_dataset("Muennighoff/flores200", flores_language, split="dev+devtest"),
        ).to_pandas(),
    )
    fleurs_df = cast(
        pd.DataFrame,
        pd.concat(
            [
                pd.DataFrame.from_records(read_fleurs(t))
                for t in Path(FLEURS_DIR).glob("*.tsv")
            ],
            axis=0,
        ),
    )
    silent_dfs = []
    for split in ("train", "dev", "test"):
        df = pd.read_table(
            silent_statistics_dir / f"{split}.tsv", sep="\t", header=None
        )
        df.columns = ["filename", "is_silent"]
        silent_dfs.append(df)
    silent_df = pd.concat(silent_dfs, axis=0)
    silent_df = silent_df.drop_duplicates(["filename", "is_silent"])
    # drop silent files
    mask = ~silent_df["is_silent"]
    silent_df = silent_df.loc[mask]
    non_silent_files = set(silent_df.filename.tolist())
    mask = fleurs_df.filename.apply(lambda file: file in non_silent_files)
    fleurs_df = fleurs_df.loc[mask]

    # Aggregate the fleurs dataset by `_id` to handle one-to-many relationships
    tuple_columns = [
        "filename",
        "gender",
        "num_samples",
        "speaker_id",
        # INFO: in the original Fleurs, same raw_transcription might occur across both splits (e.g., fleurs id 2 for af_za)
        # Therefore we need to cast `split` as a `tuple[str]`
        "split",
    ]  # replace with your column names
    # Create the aggregation dictionary
    agg_dict = {
        col: (lambda x: tuple(x)) if col in tuple_columns else "first"
        for col in fleurs_df.columns
        if col != "fleurs_id"
    }
    # Apply the groupby and aggregation
    fleurs_aggregated = fleurs_df.groupby("fleurs_id").agg(agg_dict).reset_index()
    NUM_FLEURS_SENT = len(fleurs_aggregated)

    fleurs_transcriptions: set[str] = set(
        fleurs_aggregated["raw_transcription"].tolist()
    )
    flores_transcriptions: set[str] = set(flores_df["sentence"].tolist())

    exact_matches = fleurs_transcriptions.intersection(flores_transcriptions)
    difference = fleurs_transcriptions.difference(flores_transcriptions)
    NUM_EXACT_MATCHES = len(exact_matches)
    NUM_MISSING = len(difference)

    # merge into flores so flores order is maintained for later matching with Belebele
    fleurs_flores_df = flores_df.merge(
        fleurs_aggregated, left_on="sentence", right_on="raw_transcription", how="inner"
    )
    # INFO: the below line does in rare exceptions not hold true
    # For bs_ba / bos_Latn, Flores ID 249 and 733 are the same sentence
    # assert len(fleurs_flores_df) == len(exact_matches)

    if len(difference) > 0:
        difference_: list[str] = [
            remove_punctuation(remove_extra_whitespace(line)) for line in difference
        ]
        normalized_diff_to_diff: dict[str, str] = dict(zip(difference_, difference))
        flores_transcriptions_: list[str] = [
            remove_punctuation(remove_extra_whitespace(line))
            for line in flores_transcriptions
        ]
        normalized_transcriptions_to_transcriptions: dict[str, str] = dict(
            zip(flores_transcriptions_, flores_transcriptions)
        )
        # using normalized levenshtein distance
        matches: dict[str, str] = match_sentences(
            difference_, flores_transcriptions_, LEVENSHTEIN_THRESHOLD
        )
        difference_to_match = {
            normalized_diff_to_diff[k]: normalized_transcriptions_to_transcriptions[
                matches[k]
            ]
            for k in matches
        }
        NUM_RECOVERED = len(difference_to_match)
        fleurs_aggregated["raw_transcription"] = fleurs_aggregated[
            "raw_transcription"
        ].replace(difference_to_match)
        fleurs_flores_df = flores_df.merge(
            fleurs_aggregated,
            left_on="sentence",
            right_on="raw_transcription",
            how="inner",
        )
    else:
        NUM_RECOVERED = 0
        NUM_MISSING = 0

    # validate that for every paragraph in merged, we have all flores sentences
    # we count the number of sentences per url (i.e., paragraph)
    fleurs_flores_url_counts = fleurs_flores_df.groupby("URL").count().max(1)
    flores_url_counts = (
        flores_df[
            flores_df["URL"].isin(fleurs_flores_df["URL"].unique())  # type: ignore
        ]
        .groupby("URL")
        .count()
        .max(1)
    )
    assert len(flores_url_counts) == len(fleurs_flores_url_counts)
    keep_url_mask = fleurs_flores_url_counts == flores_url_counts
    urls_to_keep = set(fleurs_flores_url_counts.loc[keep_url_mask].index)
    NUM_TOTAL_PARAGRAPHS = len(fleurs_flores_url_counts)
    NUM_RETAINED_PARAGRAPHS = len(urls_to_keep)
    NUM_MISSING_PARAGRAPHS = NUM_TOTAL_PARAGRAPHS - NUM_RETAINED_PARAGRAPHS
    fleurs_flores_df["full_paragraph"] = fleurs_flores_df["URL"].isin(urls_to_keep)  # type: ignore

    write_stats_to_csv(
        log_file,
        language,
        NUM_FLEURS_SENT,
        NUM_EXACT_MATCHES,
        NUM_MISSING,
        NUM_RECOVERED,
        NUM_TOTAL_PARAGRAPHS,
        NUM_RETAINED_PARAGRAPHS,
        NUM_MISSING_PARAGRAPHS,
    )
    return fleurs_flores_df


# TODO: merge transcription afterwards
#
def main(args):
    if args.language is None:

        def mapper(language):
            flores_language = FLEURS_TO_FLORES[language]
            lang_dataset = align(language, LOG_FILE)
            path = DATA_DIR / f"{flores_language}.parquet"
            # index=False to not store `__index_level_0__` column
            lang_dataset.to_parquet(path, index=False)
            print(f"Processed {language}")

        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            executor.map(mapper, FLEURS_TO_FLORES.keys())
    else:
        language = args.language
        flores_language = FLEURS_TO_FLORES[language]
        lang_dataset = align(language, LOG_FILE)
        path = DATA_DIR / f"{flores_language}.parquet"
        lang_dataset.to_parquet(path)


if __name__ == "__main__":
    args = parse_args()
    main(args)
