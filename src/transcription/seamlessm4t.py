import argparse
from pathlib import Path

import torch
from src.transcription.fleurs_to_seamlessm4t import FLEURSSEAMLESSM4T
from src.transcription.utils import write_to_ndjson, load_audio
from src.utils import find_project_root
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, SeamlessM4Tv2Model
from functools import partial

PROJECT = find_project_root(__file__)
DATA = PROJECT / "data"

# Initialize the model and processor
device = "cuda:0" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")
model = model.to(device)  # type: ignore
model.eval()

# for interactive evaluation
# language_code = "af_za"
# batch_size = 8
# split = "train"
# translate = False


def collate_fn(file_paths: list[Path], language_code: str) -> torch.Tensor:
    files = [load_audio(f).unsqueeze(0) for f in file_paths]
    inputs = processor(
        audios=files,
        src_lang=language_code,
        sampling_rate=16000,
        return_tensors="pt",
    )
    return inputs


def process_audio(
    language_code: str, batch_size: int, split: str = "test", translate: bool = False
):
    seamlessm4t_lang_code = FLEURSSEAMLESSM4T.get(language_code)
    if not seamlessm4t_lang_code:
        raise ValueError(f"Unsupported language code: {language_code}")

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
        collate_fn=partial(collate_fn, language_code=language_code),  # type: ignore
        num_workers=4,
    )

    result: list[str] = []
    for batch in tqdm(dataloader, desc="Encoding"):
        # Move batch to device
        batch = batch.to(device)
        # Forward pass through the model (assumes model returns embeddings)
        with torch.inference_mode():
            try:
                # Attempt to run on the default device (likely GPU)
                global model
                outputs = model.generate(
                    **batch,  # unpack input and attention_mask
                    tgt_lang=seamlessm4t_lang_code if not translate else "eng",
                    generate_speech=False,
                )
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("GPU ran out of memory. Skipping...")
                    del batch
                    torch.cuda.empty_cache()  # Clear GPU memory
                    continue
                else:
                    raise  # Re-raise the error if it's not OOM-related
        # with torch.inference_mode():
        #     outputs = model.generate(
        #         **batch,  # unpack input and attention_mask
        #         tgt_lang=seamlessm4t_lang_code if not translate else "eng",
        #         generate_speech=False,
        #     )
        translated_text_from_audio = processor.batch_decode(
            outputs[0], skip_special_tokens=True
        )
        result.extend(translated_text_from_audio)
    # Generate keyword arguments for seamlessm4t based on translation flag
    output_folder = DATA.joinpath(
        "seamlessm4t", "translation" if translate else "transcription", language_code
    )
    output_folder.mkdir(exist_ok=True, parents=True)
    output_file = output_folder.joinpath(f"{split}.jsonl")
    json_data = [
        {
            "filename": file.name,
            "seamlessm4t_asr_translation" if translate else "seamlessm4t_asr": line,
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
