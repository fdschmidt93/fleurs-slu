from language_mappings import FLEURS_TO_FLORES
from pathlib import Path
import pandas as pd
from datasets import load_dataset

from lib import get_data, match_sentences, normalized_levenshtein, remove_punctuation


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

    match_mask = merged["raw_transcription"] == merged["sentence"]
    # miss is a pd.DataFrame

    if (~match_mask).sum():
        miss = merged.loc[~match_mask]
        miss = miss.copy()
        miss.loc[:, "dist"] = miss.apply(
            lambda x: normalized_levenshtein(
                remove_punctuation(x["sentence"]),
                remove_punctuation(x["raw_transcription"]),
            ),
            axis=1,
        )
        matches = match_sentences(miss["raw_transcription"], flores["sentence"], 1)
        miss.loc[:, "dist2"] = [
            normalized_levenshtein(remove_punctuation(k), remove_punctuation(v))
            for k, v in matches.items()
        ]
        same_sum = (miss["dist"] == miss["dist2"]).sum()
        miss["lev_match"] = list(matches.values())
        bad = 0
        for row in miss.iterrows():
            # if row[1]["dist2"] < 0.05:
            print(row[1]["id"])
            print(row[1]["raw_transcription"])
            print(row[1]["sentence"], row[1]["dist"])
            print(row[1]["lev_match"], row[1]["dist2"])
            print()
            bad += 1

    else:
        matches = []
        same_sum = 0

    msg = f"{lang} - full: {len(fleurs):04} - exact: {match_mask.sum():04} - miss: {(~match_mask).sum():04} - lev same as id: {same_sum:04} "
    if match_mask.sum() == len(fleurs):
        msg += "✓"
    print(msg)


#
# for lang, lang_folder in LANGUAGE_MAPPING.items():
# main(lang, lang_folder)


# arb_Arab - full: 1702 - exact: 1535 - miss: 0167 - min lev same as id: 0164    x   3
# ceb_Latn - full: 1932 - exact: 1812 - miss: 0120 - min lev same as id: 0001    x 119
# ckb_Arab - full: 1981 - exact: 1883 - miss: 0098 - min lev same as id: 0097    x   1
# umb_Latn - full: 1493 - exact: 1116 - miss: 0377 - min lev same as id: 0327    x  50  -- text VERY often very off

lang = "arb_Arab"
lang_folder = FLEURS_TO_FLORES[lang]
