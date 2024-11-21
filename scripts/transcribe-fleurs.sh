#!/usr/bin/bash

# Validate arguments
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 model_name language_code batch_size [translate_flag]"
    exit 1
fi

# Validate model name
valid_models=("whisper" "seamlessm4t" "mms")
if [[ ! " ${valid_models[@]} " =~ " ${1} " ]]; then
    echo "Error: Model must be one of: ${valid_models[*]}"
    exit 1
fi

source "$HOME/.bashrc"
source "$HOME/miniforge3/bin/activate" fleurs-slu

project_root="/network/scratch/s/schmidtf/fleurs-slu/"
cd "${project_root}" || exit 1

# Check if the translate flag is set
if [ -n "$4" ]; then
    if python -m "src.transcription.${1}" "${2}" "${3}" --translate; then
        echo "${2}-translation completed" >> "${project_root}/logs/${1}-completed.txt"
    else
        echo "Error occurred in ${2} during translation" >> "${project_root}/logs/${1}-errors.txt"
    fi
else
    if python -m "src.transcription.${1}" "${2}" "${3}"; then
        echo "${2}-transcription completed" >> "${project_root}/logs/${1}-completed.txt"
    else
        echo "Error occurred in ${2} during transcription" >> "${project_root}/logs/${1}-errors.txt"
    fi
fi
