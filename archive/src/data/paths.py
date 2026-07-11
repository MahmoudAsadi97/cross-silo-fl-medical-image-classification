from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
FED_ISIC2019_ROOT = DATA_ROOT / "fed_isic2019"

RAW_DIR = FED_ISIC2019_ROOT / "raw"
INTERIM_DIR = FED_ISIC2019_ROOT / "interim"
PROCESSED_DIR = FED_ISIC2019_ROOT / "processed"
METADATA_DIR = FED_ISIC2019_ROOT / "metadata"
REPORTS_DIR = FED_ISIC2019_ROOT / "reports"


def ensure_data_directories() -> None:
    for path in [
        DATA_ROOT,
        FED_ISIC2019_ROOT,
        RAW_DIR,
        INTERIM_DIR,
        PROCESSED_DIR,
        METADATA_DIR,
        REPORTS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_data_directories()
    print("Project root:", PROJECT_ROOT)
    print("Dataset root:", FED_ISIC2019_ROOT)
    print("Raw dir:", RAW_DIR)
