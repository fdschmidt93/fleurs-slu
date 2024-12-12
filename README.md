# Fleurs-SLU

Fleurs-SLU links the spoken sentences from [Fleurs](https://huggingface.co/datasets/google/fleurs) to the NLU benchmarks [belebele](https://huggingface.co/datasets/facebook/belebele) and [SIB](https://huggingface.co/datasets/Davlan/sib200). This repository comprises all the required scripts to construct the two benchmarks.

The benchmarks are available at Huggingface datasets at:

- [SIB-Fleurs](https://huggingface.co/datasets/WueNLP/sib-fleurs)
- [Belebele-Fleurs](https://huggingface.co/datasets/WueNLP/belebele-fleurs)

Results for various baselines on the benchmarks are available on the respective Huggingface dataset repositories.

## Installation

Install the associated `environment.yaml` with mamba.

```
mamba create -f environment.yaml
conda activate fleurs-slu
```

## Scripts

The below scripts are the entry points to process the data and compile the datasets. They must be ran in order of the rows in the below table.

| Script                              | Description                                                                                                                                               | Data                                               | Logs                                          |
|-------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------|-----------------------------------------------|
| `download-extract-fleurs.py`        | Downloads and extracts Fleurs into `./data/fleurs`.                                                                                                       | `./data/fleurs`                                   | -                                             |
| `fleurs-remove-silence-and-noise.py`| Uses Silero-VAD with multiprocessing to detect silence and noisy samples (95% of time not corresponding to speech). Logs are used in `align-flores-and-fleurs` to remove erroneous samples. | -                                                 | `logs/fleurs-silence`                         |
| `align-flores-and-fleurs.py`        | Uses conservative Levenshtein (at most 3 characters) on normalized strings (removed punctuation, excess whitespaces, etc.) to recover some non-exact matches. | -                                                 | -                                             |
| `transcribe_translate_batch.sh`     | Transcribes (and optionally translates) with Whisper and Seamless. Requires Slurm. Check `src/transcription/fleurs_to_{whisper,seamlessm4t}.py` for language mappings. | `./data/{whisper,seamlessm4t}/{transcription,translation}` | `./logs/{whisper,seamlessm4t}/`              |
| `compute-cer-to-flores.py`          | Computes WER and CER with Huggingface `evaluate` against the original Flores sentence.                                                                     | `flores-fleurs_asr`                               | `./logs/flores-fleurs.csv`                    |
| `align-sib-fleurs.py`               | Merges sentence-aligned data from SIB-200 into the splits of FLEURS.                                                                                      | Uploaded to the Huggingface Hub                   | -                                             |
| `align-fleurs-and-belebele.py`      | Merges the sentence and paragraph-aligned data from Fleurs and Belebele                                                                                   | Uploaded to the Huggingface Hub                   | -                                             |
| `align-fleurs-sib.py`               | Merges sentence-aligned data from SIB-200 into the splits of FLEURS.                                                       | Uploaded to the Huggingface Hub                   | -                                             |

When these scripts are run, relevant derived data is created in `./data/` and logs are written to `./logs`. The logs should already comprise most information (but can surely be improved).

# Silent and Noisy File Removal

We run `$PROJECT/scripts/fleurs-remove-silence-and-noise.py`. 

We manually verified 50 samples. VAD not only removes silent files but very frequently also inaudibly noisy files. Only 
```
de_dk train 4182703406352481327.wav
```
is somewhat of a false positive. The file is still quite noisy but could be understood when listened to carefully. Please see the script for further information.

| lang   | split   |   count |
|:-------|:--------|--------:|
| nb_no  | train   |     497 |
| es_419 | train   |     490 |
| cy_gb  | train   |     394 |
| sd_in  | train   |     307 |
| ny_mw  | train   |      15 |
| ny_mw  | test    |       8 |
| ckb_iq | train   |       8 |
| wo_sn  | train   |       7 |
| ur_pk  | test    |       6 |
| ny_mw  | dev     |       6 |
| nso_za | test    |       6 |
| ps_af  | train   |       4 |
| so_so  | train   |       4 |
| fa_ir  | train   |       4 |
| ceb_ph | train   |       3 |
| lg_ug  | train   |       3 |
| kea_cv | train   |       2 |
| hy_am  | train   |       2 |
| ur_pk  | dev     |       2 |
| nso_za | dev     |       2 |
| hr_hr  | train   |       2 |
| bn_in  | train   |       2 |
| bg_bg  | train   |       2 |
| cy_gb  | test    |       2 |
| ff_sn  | train   |       2 |
| umb_ao | train   |       1 |
| ar_eg  | train   |       1 |
| en_us  | train   |       1 |
| or_in  | train   |       1 |
| da_dk  | train   |       1 |
| kn_in  | train   |       1 |
| he_il  | train   |       1 |
| kn_in  | dev     |       1 |
| he_il  | dev     |       1 |
| da_dk  | test    |       1 |
| sk_sk  | test    |       1 |
| te_in  | train   |       1 |
| ta_in  | train   |       1 |
| sk_sk  | dev     |       1 |
| he_il  | test    |       1 |
| ff_sn  | dev     |       1 |
| mn_mn  | train   |       1 |
| kn_in  | test    |       1 |
| ms_my  | train   |       1 |
| mi_nz  | dev     |       1 |
| te_in  | test    |       1 |
| ig_ng  | train   |       1 |
| kam_ke | train   |       1 |
| ha_ng  | train   |       1 |
| so_so  | test    |       1 |
