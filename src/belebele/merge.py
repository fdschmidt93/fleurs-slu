import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd
from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from src.utils import read_fleurs, remove_extra_whitespace, find_project_root
from src.levenshtein import match_sentences, normalized_levenshtein
from src.fleurs_to_flores import LANGUAGE_MAPPING


# Merge fleurs into flores
def merge_fleurs_into_flores(
    flores: pd.DataFrame, fleurs: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge the fleurs dataset into the flores dataset using exact string matches first, and fallback to Levenshtein distance.

    Args:
        flores (pd.DataFrame): The flores dataset.
        fleurs (pd.DataFrame): The fleurs dataset.

    Returns:
        pd.DataFrame: The merged dataset.
    """
    # Aggregate the fleurs dataset by `_id` to handle one-to-many relationships
    fleurs_aggregated = (
        fleurs.groupby("fleurs_id")
        .agg(lambda x: tuple(x) if not all(y == x.iloc[0] for y in x) else x.iloc[0])
        .reset_index()
    )

    fleurs_transcriptions = set(fleurs_aggregated["raw_transcription"].tolist())
    flores_transcriptions = set(flores["sentence"].tolist())

    exact_matches = fleurs_transcriptions.intersection(flores_transcriptions)
    difference = fleurs_transcriptions.difference(flores_transcriptions)

    # merge into flores so flores order is maintained for later matching with Belebele
    merged = flores.merge(
        fleurs_aggregated, left_on="sentence", right_on="raw_transcription", how="inner"
    )
    assert len(merged) == len(exact_matches)

    if len(difference) > 0:
        matches: dict[str, str] = match_sentences(
            list(difference), flores_transcriptions, 0.01
        )
        fleurs_aggregated["raw_transcription"] = fleurs_aggregated[
            "raw_transcription"
        ].replace(matches)
        merged = flores.merge(
            fleurs_aggregated,
            left_on="sentence",
            right_on="raw_transcription",
            how="inner",
        )

    # validate that for every paragraph in merged, we have all flores sentences
    keep_url = merged.groupby("URL").count().max(1) == flores[
        flores["URL"].isin(merged["URL"].unique())
    ].groupby("URL").count().max(1)
    keep_url = set(keep_url.loc[keep_url].index)
    merged = merged.loc[merged["URL"].isin(keep_url)]

    for col in ("filename", "gender", "num_samples"):
        merged[col] = merged[col].apply(lambda x: x if isinstance(x, tuple) else (x,))
    return merged


def get_ordered_sentence_indices(
    passage: str, flores_subset: pd.DataFrame
) -> Sequence[int]:
    """
    Determine the order of sentences in the subset of the flores dataset that make up the passage in belebele.

    Args:
        passage (str): The concatenated passage from belebele.
        flores_subset (pd.DataFrame): A subset of the flores dataset containing individual sentences for the given URL.

    Returns:
        Sequence[int]: A list of indices representing the order of sentences from the flores subset that match the passage.
    """
    normalized_passage = remove_extra_whitespace(passage)
    flores_sentences = flores_subset["sentence"].tolist()
    flores_ids = flores_subset["fleurs_id"].tolist()

    # given the structure of flores, this _should_ always fire
    # otherwise something must be wrong with belebele!
    flores_paragraph = " ".join(flores_sentences)
    dist = normalized_levenshtein(normalized_passage, flores_paragraph)
    assert dist < 0.05
    return list(zip(flores_ids, flores_sentences))


def merge_fleurs_into_belebele(belebele: pd.DataFrame, merged: pd.DataFrame) -> Dataset:
    """
    Match concatenated sentences from the flores dataset into the belebele dataset.

    Args:
        belebele (pd.DataFrame): The belebele dataset containing concatenated flores passages.
        flores (pd.DataFrame): The flores dataset containing individual sentences.

    Returns:
        pd.DataFrame: The belebele dataset with matched passages from flores.
    """
    # link and URL mean the same thing and match perfectly
    belebele_url: set[str] = set(belebele["link"].tolist())
    merged_url: set[str] = set(merged["URL"].tolist())
    exact_match = belebele_url.intersection(merged_url)

    # belebele_: pd.DataFrame
    belebele_ = belebele.loc[belebele["link"].isin(exact_match)]  # type: ignore
    # merged_: pd.DataFrame
    merged_ = merged.loc[merged["URL"].isin(exact_match)]  # type: ignore

    unique_passages = belebele_[["flores_passage", "link"]].drop_duplicates()

    orders = []
    for row in unique_passages.iterrows():
        orders.append(
            get_ordered_sentence_indices(
                row[1]["flores_passage"], merged[merged["URL"] == row[1]["link"]]
            )
        )

    for order in orders:
        ordered_ids, _ = zip(*order)
        rows = []
        for id_ in ordered_ids:
            mask = merged_["fleurs_id"] == id_
            subset = merged_.loc[mask]
            assert len(subset) == 1
            # list of 3-10 pd.Series in **specific** order
            rows.append(subset)
        # how to best merge in belebele_

    final_rows = []
    # Iterate through each row in unique_passages to process corresponding merged rows
    for i, (_, row) in enumerate(unique_passages.iterrows()):
        belebele_samples: list[dict] = belebele_[belebele_.link == row.link].to_dict(
            "records"
        )

        # Get the ordered IDs for the current URL
        ordered_ids, _ = zip(*orders[i])

        # Get rows from merged_ based on ordered IDs and ensure they are ordered correctly
        ordered_rows = (
            merged_.set_index("fleurs_id").loc[list(ordered_ids)].reset_index()
        ).to_dict("records")
        for i, row_ in enumerate(ordered_rows):
            row_["sentence_idx"] = i

        # Create a flattened dictionary to represent the merged data
        for sample in belebele_samples:
            sample["sentence_data"] = ordered_rows

        # Append the flattened result to the list
        final_rows.extend(belebele_samples)
    # Create a final DataFrame from the list of flattened rows
    dataset = Dataset.from_list(final_rows)
    return dataset


def main(args):
    """
    Main function to process the input datasets and merge them.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    PROJECT = find_project_root(__file__)
    FLEURS_DIR = PROJECT / "data" / "fleurs" / LANGUAGE_MAPPING[args.language]

    # Load datasets
    flores_dataset = load_dataset(
        "Muennighoff/flores200", args.language, split="dev+devtest"
    )
    belebele_dataset = load_dataset("facebook/belebele", args.language, split="test")
    # Load fleurs dataset
    fleurs_dataset = pd.concat(
        [
            pd.DataFrame.from_records(read_fleurs(t))
            for t in Path(FLEURS_DIR).glob("*.tsv")
        ],
        axis=0,
    )

    # Convert datasets to DataFrames
    flores_df = pd.DataFrame(flores_dataset)
    belebele_df = pd.DataFrame(belebele_dataset)

    # Merge the datasets
    merged_df = merge_fleurs_into_flores(flores_df, fleurs_dataset)

    # Match concatenated passages in belebele with flores
    matched_belebele_df = merge_fleurs_into_belebele(belebele_df, merged_df)

    print(f"{args.language}: {len(matched_belebele_df)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge and process Fleurs, Flores, and Belebele datasets."
    )
    parser.add_argument(
        "--language", type=str, required=True, help="Language code for the datasets."
    )
    args = parser.parse_args()

    main(args)
