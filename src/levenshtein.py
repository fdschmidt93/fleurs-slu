from typing import Optional, Sequence

from Levenshtein import distance


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


def match_sentences(
    set_A: Sequence[str], set_B: Sequence[str], min_dist: float = float("inf")
) -> dict[str, Optional[str]]:
    """
    Match sentences from set_A to the closest sentences in set_B using Levenshtein distance.

    Args:
        set_A (Sequence[str]): A sequence of sentences to be matched.
        set_B (Sequence[str]): A sequence of sentences to match against.

    Returns:
        dict[str, Optional[str]]: A dictionary where keys are sentences from set_A and values are the closest matches from set_B.
    """
    # Normalize both sets

    matches = {}
    for sentence_A in set_A:
        best_match = None
        for sentence_B in set_B:
            dist = normalized_levenshtein(sentence_A, sentence_B)
            if dist < min_dist:
                best_match = sentence_B
        if best_match is not None:
            matches[sentence_A] = best_match
    return matches
