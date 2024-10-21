#!/usr/bin/bash

#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err

source $HOME/.bashrc
source $HOME/miniforge3/bin/activate fleurs-slu
source "$(dirname "$BASH_SOURCE")/common.sh"

# ${1}: model name, one of whisper, seamlessm4t, mms
# ${2}: fleurs language code
# ${3}: batch size

# Get the current directory
project_root=$(get_project_root)

# Run the first Python script and capture the exit status
python "$project_root/src/transcription/${1}.py" "${2}" "${3}"

if [ $? -ne 0 ]; then
    echo "Error occurred in ${2} during transcription" >> "$project_root/data/logs/${1}-errors.txt"
fi

# Log completion if no errors occurred
if [ $? -eq 0 ]; then
    echo "${1}-transcription completed" >> "$project_root/data/logs/${1}-completed.txt"
fi
