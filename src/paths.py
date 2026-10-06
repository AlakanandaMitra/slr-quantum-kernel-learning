
from pathlib import Path

import os



REPO_ROOT = Path(__file__).resolve().parents[1]



DATA_DIR = Path(

    os.environ.get("SLR_DATA_DIR", REPO_ROOT / "data")

)



EMBEDDINGS_DIR = Path(

    os.environ.get("SLR_EMBEDDINGS_DIR", DATA_DIR / "embeddings")

)



RESULTS_DIR = Path(

    os.environ.get("SLR_RESULTS_DIR", REPO_ROOT / "results")

)



SPLITS_DIR = Path(

    os.environ.get("SLR_SPLITS_DIR", REPO_ROOT / "splits")

)



PUBLISHED_RESULTS_DIR = RESULTS_DIR / "published"

