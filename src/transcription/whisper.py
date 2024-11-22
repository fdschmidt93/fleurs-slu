import argparse
import warnings
from pathlib import Path
from typing import cast

import torch
from src.transcription.fleurs_to_whisper import FLEURS2WHISPER
from src.transcription.utils import write_to_ndjson, get_wav_metadata_duration
from src.utils import find_project_root
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

PROJECT = find_project_root(__file__)
DATA = PROJECT / "data" / "fleurs"

# for interactive execution
# language_code = "af_za"
# batch_size = 8
# split = "train"
# translate = False


# Initialize the model and processor
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model_id = "openai/whisper-large-v3"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)
processor = AutoProcessor.from_pretrained(model_id)
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

# Function to process audio files and save output


def process_audio(
    language_code: str, batch_size: int, split: str = "test", translate: bool = False
):
    whisper_lang_code = FLEURS2WHISPER.get(language_code)
    if not whisper_lang_code:
        raise ValueError(f"Unsupported language code: {language_code}")

    # File structure: assuming files are stored in './data/{language_code}/audio/{split}'
    audio_dir: Path = DATA / language_code / "audio" / split
    audio_files = [audio_dir.joinpath(file) for file in audio_dir.glob("*.wav")]

    if not audio_files:
        raise ValueError(f"No audio files found in {audio_dir}")

    # Generate keyword arguments for Whisper based on translation flag
    generate_kwargs: dict[str, str | bool] = {"language": whisper_lang_code}
    if translate:
        generate_kwargs["task"] = "translate"
    # Determine output folder based on translation flag
    output_folder = DATA.joinpath(
        "whisper", "translation" if translate else "transcription", language_code
    )
    output_folder.mkdir(exist_ok=True, parents=True)
    output_file = output_folder.joinpath(f"{split}.jsonl")
    # Separate files based on length: files under 30 seconds will be batched, longer files will be processed individually
    short_files: list[Path] = []
    long_files: list[Path] = []

    for file in audio_files:
        duration = get_wav_metadata_duration(file)
        if duration < 30:
            short_files.append(cast(Path, file))
        else:
            long_files.append(cast(Path, file))

    # Process short files in batches
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        result = cast(
            list[dict[str, str]],
            pipe(short_files, batch_size=batch_size, generate_kwargs=generate_kwargs),
        )
    json_data = [
        {
            "filename": file.name,
            "whisper_asr_translation" if translate else "whisper_asr": r["text"],
        }
        for file, r in zip(short_files, result)
    ]
    write_to_ndjson(json_data, output_file)

    # Process long files individually
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        generate_kwargs["return_timestamps"] = True
        result = cast(
            list[dict[str, str]],
            pipe(long_files, batch_size=1, generate_kwargs=generate_kwargs),
        )
    json_data = [
        {
            "filename": file.name,
            "whisper_asr_translation" if translate else "whisper_asr": r["text"],
        }
        for file, r in zip(long_files, result)
    ]
    write_to_ndjson(json_data, output_file)


# Set up a command line interface
def main():
    parser = argparse.ArgumentParser(
        description="Process audio files for transcription or translation."
    )
    parser.add_argument("language_code", type=str, help="Language code to process")
    parser.add_argument("batch_size", type=int, help="Batch size for processing")
    parser.add_argument(
        "--translate",
        action="store_true",
        help="If set, perform translation instead of transcription",
    )

    args = parser.parse_args()
    for split in ("train", "dev", "test"):
        process_audio(args.language_code, args.batch_size, split, args.translate)


if __name__ == "__main__":
    main()
