import os

DATASET_PATH: str = os.getenv("DATASET_PATH", "data/Brave New World_train/public")
OUTPUT_PATH: str = os.getenv("OUTPUT_PATH", "output/predictions.txt")


def set_paths(dataset_path: str, output_path: str) -> None:
    global DATASET_PATH, OUTPUT_PATH
    DATASET_PATH = dataset_path
    OUTPUT_PATH = output_path
