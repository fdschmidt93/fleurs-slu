import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.absolute()))

import urllib.request
import tarfile
from concurrent.futures import ThreadPoolExecutor
from src.language_mappings import FLEURS

URL = "https://storage.googleapis.com/xtreme_translations/FLEURS/{}.tar.gz"

# Get the current file's directory (where download_and_extract.py is located)
current_file_dir = Path(__file__).resolve().parent

# Set PROJECT_DIR as the parent of the 'scripts' directory (i.e., the project root)
PROJECT_DIR = current_file_dir.parent
output_dir = PROJECT_DIR / "data" / "fleurs"
output_dir.mkdir(exist_ok=True)


def download_and_extract(language: str):
    """Downloads and extracts a file for a specific language."""
    file_url = URL.format(language)
    file_path = output_dir / f"{language}.tar.gz"
    extract_path = output_dir / language

    # Download the file if it doesn't exist
    if not file_path.exists():
        try:
            print(f"Downloading {language}...")
            urllib.request.urlretrieve(file_url, file_path)
            print(f"Downloaded {language}")
        except Exception as e:
            print(f"Failed to download {language}: {e}")
            return
    else:
        print(f"{file_path} already exists. Skipping download.")

    # Extract the tar.gz file if not already extracted
    if not extract_path.exists():
        try:
            print(f"Extracting {language}...")
            with tarfile.open(file_path, "r:gz") as tar:
                tar.extractall(path=extract_path)
            print(f"Extracted {language}")
        except Exception as e:
            print(f"Failed to extract {language}: {e}")
    else:
        print(f"{extract_path} already exists. Skipping extraction.")


def main():
    # Use ThreadPoolExecutor to download in parallel
    with ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(download_and_extract, FLEURS)

    print("All downloads completed.")


if __name__ == "__main__":
    main()
