import sys
from pathlib import Path

# try-except for interactive repl use
try:
    sys.path.append(str(Path(__file__).parent.parent.absolute()))
except:
    pass

import pandas as pd
from src.language_mappings import FLEURS_TO_FLORES
from datasets import load_dataset
from src.utils import (
    remove_extra_whitespace,
    remove_punctuation,
    find_project_root,
)
from src.levenshtein import match_sentences
from datasets.arrow_dataset import Dataset

from concurrent.futures import ProcessPoolExecutor
from datasets import Sequence, Audio, ClassLabel, Value
from multiprocessing import Lock

write_lock = Lock()

LEVENSHTEIN_THRESHOLD = 3


try:
    PROJECT = find_project_root(__file__)
except:  # type: ignore
    # for interactive use
    FILE = str(Path("./dummy_file.py").absolute())
    PROJECT = find_project_root(FILE)
DATA_DIR = PROJECT / "data"
FLEURS_ASR_DIR = DATA_DIR / "flores-fleurs_asr"
SIB_DIR = DATA_DIR / "sib-fleurs"
SIB_DIR.mkdir(exist_ok=True)
NUM_WORKERS = 8
LOG_FILE = PROJECT / "logs" / "fleurs-sib.txt"


def write_to_log(message: str):
    with write_lock:
        with open(LOG_FILE, "a") as f:
            f.write(f"{message}\n")


def merge_flores_sib(language: str) -> None:
    flores_language = FLEURS_TO_FLORES[language]
    SIB_LANG_DIR = SIB_DIR / flores_language
    SIB_LANG_DIR.mkdir(exist_ok=True)
    for split in ("train", "validation", "test"):
        try:
            # INFO: this fleurs version already removes the
            fleurs = pd.read_parquet(FLEURS_ASR_DIR / f"{language}.parquet")
        except:
            write_to_log(f"Fleurs-ASR does not exist for {language}-{split}")
            continue
        try:
            sib = load_dataset(
                "Davlan/sib200", flores_language, split=split
            ).to_pandas()
        except:
            write_to_log(f"SIB does not exist for {language}-{split}")

        # Perform the merge with indicator=True
        merged = fleurs.merge(
            sib, left_on="sentence", right_on="text", how="right", indicator=True
        )
        # Filter for rows that only appear in 'fleurs' (i.e., no match in 'sib')
        difference = merged[merged["_merge"] == "right_only"]["text"].tolist()
        mask = merged["_merge"] == "both"
        merged = merged.loc[mask]
        del merged["_merge"]
        if len(difference) > 0:
            difference_: list[str] = [
                remove_punctuation(remove_extra_whitespace(line)) for line in difference
            ]
            normalized_diff_to_diff: dict[str, str] = dict(zip(difference_, difference))
            flores_sentence = fleurs["sentence"].tolist()
            flores_sentence_: list[str] = [
                remove_punctuation(remove_extra_whitespace(line))
                for line in flores_sentence
            ]
            normalized_sentence_to_sentence: dict[str, str] = dict(
                zip(flores_sentence_, flores_sentence)
            )
            # returns a dictionary of sentences that match within threshold
            matches: dict[str, str] = match_sentences(
                difference_, flores_sentence_, LEVENSHTEIN_THRESHOLD
            )
            difference_to_match = {
                normalized_sentence_to_sentence[matches[k]]: normalized_diff_to_diff[k]
                for k in matches
            }
            fleurs["sentence"] = fleurs["sentence"].replace(difference_to_match)
            OLD_N = len(merged)
            merged = fleurs.merge(
                sib, left_on="sentence", right_on="text", how="right", indicator=True
            )
            mask = merged["_merge"] == "both"
            merged = merged.loc[mask]
            print(f"Recovered {len(matches)}/{len(difference_)}")
            assert len(merged) >= OLD_N
            del merged["_merge"]

        path_prefix = f"./data/fleurs/{language}/audio"
        audio = [
            [
                f"{path_prefix}/{split}/{filename}"
                for split, filename in zip(splits.tolist(), filenames.tolist())
            ]
            for splits, filenames in zip(merged["split"], merged["filename"])
        ]
        merged["audio"] = audio
        dataset_ = Dataset.from_pandas(merged, preserve_index=False)
        for col in ["id", "has_image", "has_hyperlink", "fleurs_id"]:
            dataset_ = dataset_.cast_column(col, Value(dtype="int32"))
        dataset_ = dataset_.cast_column("audio", Sequence(feature=Audio(16000)))
        dataset_ = dataset_.cast_column(
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
                ],
                num_classes=7,
            ),
        )
        dataset_ = dataset_.remove_columns(["full_paragraph"])
        try:
            dataset_.push_to_hub(
                "wuenlp/fleurs-sib",
                config_name=flores_language,
                split=split,
                data_dir=f"data/{flores_language}",
            )
            write_to_log(f"{language}\t{split}\t{len(dataset_)}/{len(sib)}")
        except:
            write_to_log(f"Error uploading {language}\t{split}")
        print(f"{language}-{split}")


def main():
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        executor.map(merge_flores_sib, FLEURS_TO_FLORES.keys())


if __name__ == "__main__":
    main()
