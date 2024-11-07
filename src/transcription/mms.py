import argparse
from functools import partial
from pathlib import Path

import torch
from src.transcription.fleurs_to_mms import FLEURS2MMS
from src.utils import find_project_root
from src.transcription.utils import write_to_ndjson, load_audio
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, Wav2Vec2ForCTC


PROJECT = find_project_root(__file__)
# PROJECT = find_project_root("./src/transcription/mms.py")
DATA = PROJECT / "data"

# Let's create the mapping from Fleurs language codes to SeamlessM4T language codes.
# Where there's no direct match, we'll map to the closest available language in SeamlessM4T.


model_id = "facebook/mms-1b-l1107"
# Initialize the model and processor
device = "cuda:0" if torch.cuda.is_available() else "cpu"


def collate_fn(file_paths: Path | list[Path], processor) -> torch.Tensor:
    file_path = file_paths[0] if isinstance(file_paths, list) else file_paths
    file = load_audio(file_path)
    return processor(file, sampling_rate=16_000, return_tensors="pt")


def process_audio(
    language_code: str, batch_size: int, split: str = "test", translate: bool = False
):
    mms_lang_code = FLEURS2MMS.get(language_code)
    if not mms_lang_code:
        raise ValueError(f"Unsupported language code: {language_code}")

    processor = AutoProcessor.from_pretrained(model_id, target_lang=mms_lang_code)
    model = Wav2Vec2ForCTC.from_pretrained(
        model_id, target_lang=mms_lang_code, ignore_mismatched_sizes=True
    )
    model.to(device)

    # File structure: assuming files are stored in './data/{language_code}/audio/{split}'
    audio_dir = DATA / "fleurs" / language_code / "audio" / split
    audio_files: list[Path] = [audio_dir / file for file in audio_dir.glob("*.wav")]
    if not audio_files:
        raise ValueError(f"No audio files found in {audio_dir}")

    dataloader = DataLoader(
        audio_files,  # type: ignore
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        collate_fn=partial(collate_fn, processor=processor),  # type: ignore
        num_workers=4,
    )

    result: list[str] = []
    for batch in tqdm(dataloader, desc="Encoding"):
        # Move batch to device
        batch = batch.to(device)
        # Forward pass through the model (assumes model returns embeddings)
        with torch.inference_mode():
            outputs = model(
                **batch,  # unpack input and attention_mask
            )
        ids = torch.argmax(outputs.logits, dim=-1)[0]
        translated_text_from_audio = processor.decode(ids)
        result.append(translated_text_from_audio)
    # Generate keyword arguments for seamlessm4t based on translation flag
    output_folder = DATA.joinpath(
        "mms", "translation" if translate else "transcription", language_code
    )
    output_folder.mkdir(exist_ok=True, parents=True)
    output_file = output_folder.joinpath(f"{split}.jsonl")
    json_data = [
        {
            "filename": file.name,
            "mms_asr_translation" if translate else "mms_asr": line,
        }
        for file, line in zip(audio_files, result)
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
