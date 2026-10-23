from huggingface_hub import hf_hub_download
import shutil
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from common.const import DATASET, HF_CATEGORIES_FILE, LOCAL_CATEGORIES_FILE_PATH

file_path = hf_hub_download(
    repo_id=DATASET,
    filename=HF_CATEGORIES_FILE,
    repo_type="dataset"
)

shutil.copy(file_path, LOCAL_CATEGORIES_FILE_PATH)