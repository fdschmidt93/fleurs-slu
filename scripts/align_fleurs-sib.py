import sys
from pathlib import Path

# sys.path.append(str(Path(__file__).parent.parent.absolute()))

import pandas as pd
from src.sib.langs import SIB_LANGS
from src.language_mappings import FLEURS_TO_FLORES
from pathlib import Path
from datasets import load_dataset
from src.utils import (
    read_fleurs,
    remove_extra_whitespace,
    remove_punctuation,
    find_project_root,
)
from src.levenshtein import match_sentences
from datasets.arrow_dataset import Dataset

from datasets import Sequence, Audio, ClassLabel, Value
# Merge fleurs into SIB

# load asr flores
# load sib
# merge on sentence
# profit

LEVENSHTEIN_THRESHOLD = 3


# PROJECT = find_project_root(__file__)
FILE = str(Path("./create_fleurs_belebele.py").absolute())
PROJECT = find_project_root(FILE)
DATA_DIR = PROJECT / "data"
FLEURS_ASR_DIR = DATA_DIR / "flores-fleurs_asr"
SIB_DIR = DATA_DIR / "sib-fleurs"
SIB_DIR.mkdir(exist_ok=True)

languages = [
#         ("ne_np", "npi_Deva"),
# ("bg_bg", "bul_Cyrl"),
("ckb_iq", "ckb_Arab"),
("oc_fr", "oci_Latn"),
("sd_in", "snd_Arab"),

    ]
# for (language, flores_language) in FLEURS_TO_FLORES.items():
for (language, flores_language) in languages:
    SIB_LANG_DIR = SIB_DIR / flores_language
    SIB_LANG_DIR.mkdir(exist_ok=True)
    for split in ("train", "validation", "test"):
        try:
            fleurs = pd.read_parquet(FLEURS_ASR_DIR / f"{language}.parquet")
        except:
            continue
        sib = load_dataset("Davlan/sib200", flores_language, split=split).to_pandas()

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

        dataset_.remove_columns(["full_paragraph"])
        dataset_.push_to_hub(
            "wuenlp/sib",
            config_name=flores_language,
            split=split,
            data_dir=f"data/{flores_language}",
        )
        # merged.to_parquet(SIB_LANG_DIR / f"{split}.parquet", index=False)
        print(f"{language}-{split}")

# merged_ = merged.copy()
#
# # add split and filename
# path_prefix = "./data/fleurs/de_de/audio"
# audio = [
#     [
#         f"{path_prefix}/{split}/{filename}"
#         for split, filename in zip(splits.tolist(), filenames.tolist())
#     ]
#     for splits, filenames in zip(merged_["split"], merged_["filename"])
# ]
# merged_["audio"] = audio
#
# dataset_ = Dataset.from_pandas(merged_, preserve_index=False)
#
# {'entertainment',
#  'geography',
#  'health',
#  'politics',
#  'science/technology',
#  'sports',
#  'travel'}
#
# dataset_ = dataset_.cast_column("audio", Sequence(feature=Audio(16000)))
#
# dataset___ = dataset_.remove_columns(
#     [
#         "URL",
#         "id",
#         "domain",
#         "topic",
#         "has_image",
#         "has_hyperlink",
#         "fleurs_id",
#         "filename",
#         "raw_transcription",
#         "transcription",
#         "num_samples",
#         "speaker_id",
#         "gender",
#         "split",
#         "full_paragraph",
#         "whisper_asr",
#         "whisper_asr_cer",
#         "whisper_asr_wer",
#         "whisper_asr_translation",
#         "seamlessm4t_asr",
#         "seamlessm4t_asr_cer",
#         "seamlessm4t_asr_wer",
#         "seamlessm4t_asr_translation",
#         "index_id",
#         "category",
#         "text",
#     ]
# )
# dataset___ = dataset___.select(range(10))
# dataset___.push_to_hub(
#     "fdschmidt93/Test", config_name="test", split="train", data_dir="data"
# )
#
#
# splits_filenames = [
#     (splits, filenames)
#     for splits, filenames in zip(merged_["split"l], merged_["filename"])
# ]
