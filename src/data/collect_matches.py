from pathlib import Path

import kagglehub

from src.config import RAW_DATA_DIR


UCL_KAGGLE_DATASET = "raminvali/uefa-champions-league"


def download_historical_ucl_data() -> Path:
    """
    Download the historical UEFA Champions League dataset from Kaggle.

    Returns
    -------
    Path
        Local directory containing the downloaded dataset.
    """

    output_dir = RAW_DATA_DIR / "matches" / "ucl_kaggle"

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 60)
    print("UCL PREDICTOR - HISTORICAL MATCH DATA")
    print("=" * 60)

    print("\nDownloading historical Champions League dataset...")
    print(f"Dataset: {UCL_KAGGLE_DATASET}")
    print(f"Destination: {output_dir}")

    downloaded_path = kagglehub.dataset_download(
        UCL_KAGGLE_DATASET,
        output_dir=str(output_dir),
    )

    print("\nDownload complete.")
    print(f"Dataset location: {downloaded_path}")

    return Path(downloaded_path)


if __name__ == "__main__":
    download_historical_ucl_data()