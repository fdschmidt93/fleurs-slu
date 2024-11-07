import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.absolute()))

from evaluate import load


import pandas as pd
from src.language_mappings import FLEURS_TO_FLORES
from src.utils import (
    find_project_root,
)
from src.language_mappings import FLEURS
from concurrent.futures import ThreadPoolExecutor


# PROJECT = find_project_root(__file__)
FILE = str(Path("./create_fleurs_belebele.py").absolute())
PROJECT = find_project_root(FILE)
LOG_FILE = PROJECT / "logs" / "flores-fleurs.csv"
DATA_DIR = PROJECT / "data"
DATA_DIR.mkdir(exist_ok=True)
FLEURS_ASR_DIR = PROJECT / "data" / "flores-fleurs_asr"
FLEURS_ASR_DIR.mkdir(exist_ok=True)


def process_lang(lang: str):
    print(f"Processing {lang}")
    from datetime import datetime

    # experiment_id resolves deadlocks in concurrent processes
    microtimestamp = int(datetime.timestamp(datetime.now()) * 1e6)
    metrics = {
            "cer": load("cer", experiment_id=str(microtimestamp), keep_in_memory=True),
            "wer": load("wer", experiment_id=str(microtimestamp), keep_in_memory=True),
            }
    fleurs_flores = pd.read_parquet(
            DATA_DIR / "flores-fleurs_raw" / f"{FLEURS_TO_FLORES[lang]}.parquet"
            )
    fleurs_flores.loc[:, ["sentence", "filename"]].head()
    original_count = fleurs_flores.groupby(["URL"]).count().max(1)

    fleurs_flores_unrolled = fleurs_flores.explode(
            ["filename", "num_samples", "speaker_id", "gender", "split"]
            ).reset_index(drop=True)
    fleurs_flores_unrolled["sentence"] = fleurs_flores_unrolled["sentence"].apply(
            lambda x: x.strip()
            )
    COMPLETE = True
    for model in ("whisper", "seamlessm4t"):
        for task in ("transcription", "translation"):
            if COMPLETE:
                try:
                    model_lang_dir = DATA_DIR / model / task / lang
                    model_transcriptions = []
                    for split in ("train", "dev", "test"):
                        transcription_path = model_lang_dir / f"{split}.jsonl"
                        if not transcription_path:
                            print(f"{transcription_path} does not exist")
                        transcription = pd.read_json(transcription_path, lines=True)
                        columns = [
                                c
                                for c in transcription.columns
                                if any(x in c for x in ("filename", "asr"))
                                ]
                        assert len(columns) == 2
                        transcription = transcription.loc[:, columns]
                        for col in columns:
                            transcription[col] = transcription[col].apply(
                                    lambda x: x.strip()
                                    )
                        model_transcriptions.append(transcription)
                    transcription = pd.concat(model_transcriptions, axis=0)
                    fleurs_flores_unrolled = fleurs_flores_unrolled.merge(
                            transcription, on="filename", how="inner"
                            )
                    # assert (
                            #     len(transcription)
                            #     == fleurs_flores_unrolled["filename"].nunique()
                            # )
                    if task == "transcription":
                        # using _compute is so much faster, HF weirdness
                        for key, metric in metrics.items():
                            results = []
                            for pred, ref in zip(
                                    fleurs_flores_unrolled[f"{model}_asr"],
                                    fleurs_flores_unrolled["sentence"],
                                    ):
                                results.append(
                                        metric._compute(
                                            predictions=[pred], references=[ref]
                                            )
                                        )
                            fleurs_flores_unrolled[f"{model}_asr_{key}"] = results
                except:
                    print(f"Error with {lang}-{model}-{task}")

    if COMPLETE:
        # Define the list of columns you want to cast to tuples
        tuple_columns = [
                "filename",
                "gender",
                "num_samples",
                "speaker_id",
                "split",
                "whisper_asr",
                "whisper_asr_cer",
                "whisper_asr_wer",
                "whisper_asr_translation",
                "seamlessm4t_asr",
                "seamlessm4t_asr_cer",
                "seamlessm4t_asr_wer",
                "seamlessm4t_asr_translation",
                ]  # replace with your column names

        # Create the aggregation dictionary
        agg_dict = {
                col: (lambda x: tuple(x)) if col in tuple_columns else "first"
                for col in fleurs_flores_unrolled.columns
                if col not in ["sentence", "URL"]  # Exclude the grouping columns
                }

        # Apply the groupby and aggregation
        fleurs_flores_unrolled_ = (
                fleurs_flores_unrolled.groupby(["sentence", "URL"])
                .agg(agg_dict)
                .reset_index()
                )
        new_count = fleurs_flores_unrolled_.groupby(["URL"]).count().max(1)
        mask = original_count == new_count
        fleurs_flores_unrolled_ = fleurs_flores_unrolled_.loc[
                mask.loc[fleurs_flores_unrolled_["URL"]].reset_index()[0]
                ]
        fleurs_flores_unrolled_.to_parquet(
                FLEURS_ASR_DIR.joinpath(f"{lang}.parquet"), index=False
                )
        print(f"Completed {lang}")


with ThreadPoolExecutor(max_workers=16) as executor:
    executor.map(process_lang, FLEURS)
#
# for lang in FLEURS:
#     process_lang(lang)
