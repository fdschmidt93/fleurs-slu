#!/bin/bash

# Function to determine the project root, whether you're in the project root or scripts directory
get_project_root() {
    # Get the current directory
    local current_dir=$(pwd)

    # If we are in "scripts", the project root is one directory up
    if [[ "$(basename "$current_dir")" == "scripts" ]]; then
        echo "$(dirname "$current_dir")"
    else
        # Otherwise, assume we are already in the project root
        echo "$current_dir"
    fi
}
