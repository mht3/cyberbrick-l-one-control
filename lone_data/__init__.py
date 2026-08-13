"""L-ONE data collection.

Datasets live in a local folder and never touch the Hugging Face Hub -- not to
pull, not to push. LeRobot otherwise treats a repo_id as a Hub coordinate and
will try to download a dataset whose local files look incomplete, which turns a
missing directory into a 401 credentials prompt.

These flags are read by huggingface_hub/datasets at import time, so they have to
be set before anything imports lerobot -- importing this package is what
guarantees that for every entry point in the repo.
"""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
