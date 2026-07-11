# Data — Fed-ISIC2019

The real dataset is **gitignored**. Only the tiny synthetic fixture
(`data/fixtures/fed_isic2019_tiny/`) is committed, so tests and the smoke tier run
without it.

## Expected on-disk layout
```
data/fed_isic2019/
  raw/
    train/client_<0..5>/class_<0..7>/*.jpg
    test/ client_<0..5>/class_<0..7>/*.jpg
  metadata/            # client_sizes.csv, label_mapping.json, ...
  reports/             # heterogeneity CSVs
```
`class_<j>` folder index == label index; class names are in `fl_med.CLASS_NAMES`.

## How to obtain it
**Option A — FLamby (authoritative).** In an isolated env (FLamby pins older deps):
```bash
git clone https://github.com/owkin/FLamby && cd FLamby
pip install -e ".[isic2019]"
python flamby/datasets/fed_isic2019/dataset_creation_scripts/download_isic.py \
    --output-folder /path/to/ISIC_2019
# then preprocess/resize per FLamby's scripts, and arrange into the raw/ layout above.
```
**Option B — Hugging Face mirror** (`flwrlabs/fed-isic2019`): same official splits;
export into the `raw/train|test/client_*/class_*` layout. A helper lives at
`scripts/setup/download_hf_fed_isic2019.py` (legacy; verify before use).

Keep the **official split** either way so results stay comparable to the leaderboard.

## Split note
`metadata/client_sizes.csv` defines a train/val/test split. On disk `raw/train` holds
the **train** portion and `raw/test` the **test** portion; the **val** portion is not
materialized as folders. Totals: train 14,888 + val 3,709 + test 4,650 = 23,247.
