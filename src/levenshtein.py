from typing import Sequence
from Levenshtein import distance


def match_sentences(
    set_A: Sequence[str],
    set_B: Sequence[str] | set[str] | tuple[str, ...],
    max_dist: float = float("inf"),
) -> dict[str, str]:
    """
    Match sentences from set_A to the closest sentences in set_B using Levenshtein distance.

    Args:
        set_A (Sequence[str]): A sequence of sentences to be matched.
        set_B (Sequence[str]): A sequence of sentences to match against.
        max_dist (float): Maximum Levenshtein distance allowed for a match. Defaults to infinity.

    Returns:
        dict[str, str]: A dictionary where keys are sentences from set_A and values are the closest matches from set_B.
    """
    matches = {}
    for sentence_A in set_A:
        best_match = None
        min_dist = float("inf")  # Track the closest match for the current sentence_A
        for sentence_B in set_B:
            dist = distance(sentence_A, sentence_B)
            if dist < min_dist and dist <= max_dist:
                best_match = sentence_B
                min_dist = dist  # Update the minimum distance found for this sentence
        if best_match is not None:
            matches[sentence_A] = best_match
    return matches


def normalized_levenshtein(a: str, b: str) -> float:
    """
    Calculate the normalized Levenshtein distance between two strings.

    Args:
        a (str): The first string.
        b (str): The second string.

    Returns:
        float: The normalized Levenshtein distance between the two strings.
    """
    return distance(a, b) / max(len(a), len(b))
