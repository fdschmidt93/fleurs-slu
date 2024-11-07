from language_mappings import FLEURS_TO_FLORES
from pathlib import Path
import pandas as pd
from datasets import load_dataset

from lib import get_data, match_sentences


def main(lang, lang_folder):
    # Load fleurs dataset
    cwd = Path.cwd()
    paths = ["fleurs", "data", lang_folder]
    fleurs = pd.concat(
        [get_data(t) for t in cwd.joinpath(*paths).glob("*.tsv")],
        axis=0,
    )
    fleurs["fleurs_id"] = fleurs["fleurs_id"].astype(int)
    fleurs = fleurs = (
        fleurs.groupby("fleurs_id")
        .agg(lambda x: tuple(x) if not all(y == x.iloc[0] for y in x) else x.iloc[0])
        .reset_index()
    )
    # Convert flores dataset to DataFrame
    flores = pd.DataFrame(
        load_dataset("Muennighoff/flores200", lang, split="dev+devtest")
    )
    flores["idx"] = flores.index
    ids = pd.read_json("./flores_idx_fleurs_id.json")
    flores = flores.merge(ids, left_on="idx", right_on="idx", how="inner")
    flores["fleurs_id"] = flores["fleurs_id"].astype(int)
    merged = fleurs.merge(
        flores, left_on="fleurs_id", right_on="fleurs_id", how="inner"
    )

    fleurs_transcriptions = set(merged["raw_transcription"].tolist())
    flores_transcriptions = set(merged["sentence"].tolist())

    exact_matches = fleurs_transcriptions.intersection(flores_transcriptions)
    difference = fleurs_transcriptions.difference(flores_transcriptions)

    if len(difference) > 0:
        matches = match_sentences(list(difference), flores_transcriptions, 0.025)
        matches = {k: v for k, v in matches.items() if v is not None}
    else:
        matches = []
    msg = f"{lang} - full: {len(fleurs):04} - exact: {len(exact_matches):04} - miss: {len(difference):04} - lev: {len(matches):04} "
    if len(exact_matches) == len(fleurs):
        msg += "✓"
    print(msg)


for lang, lang_folder in FLEURS_TO_FLORES.items():
    main(lang, lang_folder)


# lang = "arb_Arab"
# lang_folder = LANGUAGE_MAPPING[lang]
