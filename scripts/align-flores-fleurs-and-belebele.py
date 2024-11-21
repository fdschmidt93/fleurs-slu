import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.absolute()))

from concurrent.futures import ThreadPoolExecutor
from src.language_mappings import FLEURS_TO_FLORES, BELEBELE
from src.utils import find_project_root
from src.belebele import merge_fleurs_into_belebele
import threading
from datasets.load import load_dataset
from datasets.arrow_dataset import Dataset
from typing import cast
import pandas as pd
import argparse
import csv
from datasets import Audio, Sequence

# PROJECT = find_project_root(__file__)
# from pathlib import Path

FILE = str(Path("./create_fleurs_belebele.py").absolute())
PROJECT = find_project_root(FILE)
LOG_FILE = PROJECT / "logs" / "belebele_aligned.csv"
LOG_ERROR_DIR = PROJECT / "logs" / "belebele_aligned_error"
BELEBELE_DATA_DIR = PROJECT / "data" / "belebele_asr"

write_lock = threading.Lock()
error_lock = threading.Lock()

uploaded = {
    "afr_Latn",
    "amh_Ethi",
    "arb_Arab",
    "asm_Beng",
    "azj_Latn",
    "ben_Beng",
    "bul_Cyrl",
    "cat_Latn",
    "ceb_Latn",
    "ces_Latn",
    "ckb_Arab",
    "dan_Latn",
    "deu_Latn",
    "ell_Grek",
    "eng_Latn",
    "est_Latn",
    "fin_Latn",
    "fra_Latn",
    "guj_Gujr",
    "hau_Latn",
    "heb_Hebr",
    "hin_Deva",
    "hrv_Latn",
    "hun_Latn",
    "hye_Armn",
    "ibo_Latn",
    "ind_Latn",
    "isl_Latn",
    "ita_Latn",
    "jav_Latn",
    "jpn_Jpan",
    "kan_Knda",
    "kat_Geor",
    "kaz_Cyrl",
    "kea_Latn",
    "khm_Khmr",
    "kir_Cyrl",
    "kor_Hang",
    "lao_Laoo",
    "lit_Latn",
    "lug_Latn",
    "luo_Latn",
    "lvs_Latn",
    "mkd_Cyrl",
    "pes_Arab",
    "spa_Latn",
    "tgl_Latn",
    "zho_Hans",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge and process Fleurs, Flores, and Belebele datasets."
    )
    parser.add_argument(
        "--language",
        type=str,
        help="Fleurs language code for the datasets. Use 'all' for all languages.",
    )
    args = parser.parse_args()
    return args


def write_stats_to_csv(file_path: Path, language: str, retained_paragraphs: int):
    """
    Write statistics to a CSV file. Creates a new file with headers if it doesn't exist,
    otherwise appends the new row to the existing file.
    """
    headers = [
        "Language",
        "Retained Paragraphs",
    ]

    row = [language, retained_paragraphs]

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


def align(language: str) -> Dataset:
    """
    Main function to process the input datasets and merge them.

    """
    flores_language = FLEURS_TO_FLORES[language]

    # Load datasets
    path = PROJECT / "data" / "flores-fleurs_asr" / f"{language}.parquet"
    fleurs_flores_df = cast(
        pd.DataFrame,
        cast(
            Dataset, load_dataset("parquet", data_files=str(path), split="train")
        ).to_pandas(),
    )
    fleurs_flores_df = fleurs_flores_df.loc[fleurs_flores_df.full_paragraph == True]
    belebele_df = cast(
        Dataset, load_dataset("facebook/belebele", flores_language, split="test")
    ).to_pandas()

    # Match concatenated passages in belebele with flores
    matched_belebele_dataset = merge_fleurs_into_belebele(belebele_df, fleurs_flores_df)

    for line in matched_belebele_dataset:
        for sent in line["sentence_data"]:
            splits = sent["split"]
            filenames = sent["filename"]
            paths = [
                f"/network/scratch/s/schmidtf/fleurs-slu/data/fleurs/{language}/audio/{split}/{filename}"
                for filename, split in zip(filenames, splits)
            ]
            for path in paths:
                assert Path(path).exists()
            sent["audio"] = [
                {
                    "path": f"/network/scratch/s/schmidtf/fleurs-slu/data/fleurs/{language}/audio/{split}/{filename}",
                    "sampling_rate": 16000,
                }
                for filename, split in zip(filenames, splits)
            ]
    dataset_ = Dataset.from_list(matched_belebele_dataset)
    feat = dataset_.features.copy()
    feat["sentence_data"][0]["audio"] = Sequence(Audio(sampling_rate=16000))
    dataset_ = dataset_.cast(feat)
    return dataset_


def main(args):
    def mapper(language: str):
        flores_language = FLEURS_TO_FLORES[language]
        if flores_language in BELEBELE:
            if flores_language not in uploaded:
                lang_dataset = align(language)
                # path = BELEBELE_DATA_DIR / f"{flores_language}.parquet"
                # lang_dataset.to_parquet(path)
                try:
                    lang_dataset.push_to_hub(
                        "wuenlp/belebele",
                        config_name=flores_language,
                        split="test",
                        data_dir=f"data/{flores_language}",
                    )
                    print(
                        f"{language} comprises {len(lang_dataset)} of 900 Belebele rows."
                    )
                    write_stats_to_csv(LOG_FILE, language, len(lang_dataset))
                except Exception as e:
                    # Capture the error message as a string
                    # LOG_ERROR_dir is a pathlib.Path
                    print(language, "error")
                    print(str(e))
                    with open(LOG_ERROR_DIR.joinpath(f"{language}.txt"), "w") as file:
                        file.write(str(e) + "\n")
        else:
            print(f"{language} | {flores_language} not in Belebele.")

    if args.language is None:
        with ThreadPoolExecutor(max_workers=4) as executor:
            executor.map(mapper, FLEURS_TO_FLORES.keys())
    else:
        mapper(args.language)


if __name__ == "__main__":
    args = parse_args()
    main(args)
