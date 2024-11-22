from typing import Sequence

import pandas as pd
from src.utils import (
    remove_extra_whitespace,
    remove_punctuation,
)
from src.levenshtein import normalized_levenshtein


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
    normalized_passage = remove_punctuation(remove_extra_whitespace(passage))
    flores_subset_ = flores_subset.sort_values(["id"])
    flores_sentences = flores_subset_["sentence"].tolist()
    flores_ids = flores_subset_["fleurs_id"].tolist()

    # given the structure of flores, this _should_ always fire
    # otherwise something must be wrong with belebele!
    flores_paragraph = " ".join(
        [remove_punctuation(remove_extra_whitespace(s)) for s in flores_sentences]
    )
    dist = normalized_levenshtein(normalized_passage, flores_paragraph)
    # INFO: this is a very conservative hurdle which all samples should pass
    #       not passing means most likely a sentence is missing
    assert dist < 0.05
    return flores_ids


def merge_fleurs_into_belebele(belebele: pd.DataFrame, merged: pd.DataFrame) -> list:
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

    final_rows = []
    for i, (_, row) in enumerate(unique_passages.iterrows()):
        ordered_fleurs_ids = get_ordered_sentence_indices(
            row["flores_passage"], merged.loc[merged["URL"] == row["link"]]
        )
        belebele_samples: list[dict] = belebele_[belebele_.link == row.link].to_dict(
            "records"
        )
        # Get rows from merged_ based on ordered IDs and ensure they are ordered correctly
        # INFO: the same fleurs id may have multiple rows, so we need to check URL as well
        mask = (merged_["fleurs_id"].isin(ordered_fleurs_ids)) & (
            merged_["URL"] == row["link"]
        )
        ordered_rows = merged_.loc[mask].to_dict("records")
        assert len(ordered_rows) == mask.sum().item()
        for i, row_ in enumerate(ordered_rows):
            row_["sentence_idx"] = i

        # Create a flattened dictionary to represent the merged data
        for sample in belebele_samples:
            sample["sentence_data"] = tuple(ordered_rows)

        # Append the flattened result to the list
        final_rows.extend(belebele_samples)
    # Create a final DataFrame from the list of flattened rows
    # dataset = Dataset.from_list(final_rows)
    return final_rows
