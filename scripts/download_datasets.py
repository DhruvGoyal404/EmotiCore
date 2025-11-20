"""
Dataset Download Script for Multimodal Emotion Detection System
Downloads FER-2013 and text emotion datasets from Kaggle
"""

import os
import sys
import zipfile
import shutil
from pathlib import Path

def setup_kaggle_credentials():
    """Setup Kaggle API credentials"""
    # Check for kaggle.json in various locations
    possible_paths = [
        Path('.kaggle/kaggle.json'),  # Project root
        Path.home() / '.kaggle' / 'kaggle.json',  # User home
        Path(os.environ.get('KAGGLE_CONFIG_DIR', '')) / 'kaggle.json'
    ]

    kaggle_json_found = None
    for path in possible_paths:
        if path.exists():
            kaggle_json_found = path
            break

    if kaggle_json_found:
        # Ensure it's in the default location
        default_kaggle_dir = Path.home() / '.kaggle'
        default_kaggle_json = default_kaggle_dir / 'kaggle.json'

        if not default_kaggle_json.exists():
            default_kaggle_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(kaggle_json_found, default_kaggle_json)
            # Set permissions (Unix only)
            if os.name != 'nt':
                os.chmod(default_kaggle_json, 0o600)

        print(f"Kaggle credentials found at: {kaggle_json_found}")
        return True
    else:
        print("ERROR: kaggle.json not found!")
        print("Please place your kaggle.json in one of these locations:")
        for path in possible_paths:
            print(f"  - {path}")
        print("\nTo get kaggle.json:")
        print("1. Go to kaggle.com -> Account -> Create New API Token")
        print("2. Download and place kaggle.json in the correct location")
        return False

def download_fer2013():
    """Download FER-2013 facial emotion dataset"""
    from kaggle.api.kaggle_api_extended import KaggleApi

    print("\n" + "="*50)
    print("Downloading FER-2013 Dataset...")
    print("="*50)

    api = KaggleApi()
    api.authenticate()

    # Create data directory
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)

    # Download dataset
    try:
        # Try the competition dataset first
        print("Attempting to download from competition...")
        api.competition_download_files(
            'challenges-in-representation-learning-facial-expression-recognition-challenge',
            path=str(data_dir)
        )
    except Exception as e:
        print(f"Competition download failed: {e}")
        print("Trying alternative dataset...")
        try:
            # Alternative: msambare/fer2013
            api.dataset_download_files(
                'msambare/fer2013',
                path=str(data_dir)
            )
        except Exception as e2:
            print(f"Alternative download also failed: {e2}")
            print("\nManual download instructions:")
            print("1. Go to: https://www.kaggle.com/datasets/msambare/fer2013")
            print("2. Download and extract to 'data/' folder")
            return False

    # Extract zip files
    for zip_file in data_dir.glob('*.zip'):
        print(f"Extracting {zip_file}...")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        zip_file.unlink()  # Remove zip after extraction

    print("FER-2013 dataset downloaded successfully!")
    return True

def download_text_emotion():
    """Download text emotion dataset"""
    from kaggle.api.kaggle_api_extended import KaggleApi

    print("\n" + "="*50)
    print("Downloading Text Emotion Dataset...")
    print("="*50)

    api = KaggleApi()
    api.authenticate()

    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)

    try:
        # Try different possible dataset names
        datasets_to_try = [
            'praveengovi/emotions-dataset-for-nlp',
            'pashupatigupta/emotion-detection-from-text',
            'ishantjuyal/emotions-in-text'
        ]

        downloaded = False
        for dataset in datasets_to_try:
            try:
                print(f"Trying dataset: {dataset}")
                api.dataset_download_files(dataset, path=str(data_dir))
                downloaded = True
                break
            except Exception:
                continue

        if not downloaded:
            print("Could not find text emotion dataset automatically.")
            print("\nManual download instructions:")
            print("1. Search for 'emotion text dataset' on Kaggle")
            print("2. Download and rename to 'emotion_sentimen_dataset.csv'")
            print("3. Place in 'data/' folder")
            return False

        # Extract zip files
        for zip_file in data_dir.glob('*.zip'):
            print(f"Extracting {zip_file}...")
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            zip_file.unlink()

        print("Text emotion dataset downloaded successfully!")
        return True

    except Exception as e:
        print(f"Error downloading text emotion dataset: {e}")
        return False

def verify_datasets():
    """Verify that datasets are downloaded correctly"""
    print("\n" + "="*50)
    print("Verifying Datasets...")
    print("="*50)

    data_dir = Path('data')

    # Check for FER-2013
    fer_files = list(data_dir.rglob('fer2013.csv')) + list(data_dir.rglob('*.csv'))
    if fer_files:
        print(f"Found facial emotion data: {fer_files[0]}")
    else:
        print("WARNING: FER-2013 CSV not found")

    # Check for train/test folders (alternative FER format)
    train_dir = data_dir / 'train'
    test_dir = data_dir / 'test'
    if train_dir.exists() and test_dir.exists():
        print(f"Found image folders: train/ and test/")

    # List all files in data directory
    print("\nFiles in data directory:")
    for f in data_dir.rglob('*'):
        if f.is_file():
            size = f.stat().st_size / (1024 * 1024)  # Size in MB
            print(f"  {f.relative_to(data_dir)} ({size:.2f} MB)")

def main():
    """Main function to download all datasets"""
    print("="*50)
    print("Multimodal Emotion Detection - Dataset Downloader")
    print("="*50)

    # Setup credentials
    if not setup_kaggle_credentials():
        sys.exit(1)

    # Download datasets
    fer_success = download_fer2013()
    text_success = download_text_emotion()

    # Verify
    verify_datasets()

    # Summary
    print("\n" + "="*50)
    print("Download Summary")
    print("="*50)
    print(f"FER-2013 (Facial): {'SUCCESS' if fer_success else 'FAILED'}")
    print(f"Text Emotion: {'SUCCESS' if text_success else 'FAILED'}")

    if fer_success and text_success:
        print("\nAll datasets downloaded successfully!")
        print("\nNext steps:")
        print("1. Train the 7-emotion model: python scripts/train_7_emotions.py")
        print("2. Run the API: uvicorn api.main:app --reload")
    else:
        print("\nSome datasets failed to download.")
        print("Please download them manually from Kaggle.")

if __name__ == "__main__":
    main()
