import os

DATASET_PATH: str = os.getenv("DATASET_PATH")
OUTPUT_PATH: str = os.getenv("OUTPUT_PATH")

def set_paths(dataset_path: str | None, output_path: str | None) -> None:
    global DATASET_PATH, OUTPUT_PATH
    if dataset_path is not None:
        DATASET_PATH = dataset_path
    if output_path is not None:
        OUTPUT_PATH = output_path
