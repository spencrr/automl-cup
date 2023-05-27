import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from common import get_logger
from dataset import AutoMLCupDataset
from sklearn.metrics import accuracy_score

VERBOSITY_LEVEL = "INFO"
LOGGER = get_logger(VERBOSITY_LEVEL, __file__)


def parse_args():
    default_app_dir = Path("/app/")
    default_input_dir = default_app_dir / "input"
    default_output_data_dir = default_app_dir / "output"
    default_input_data_dir = default_app_dir / "input_data"

    parser = ArgumentParser()
    parser.add_argument("--input_dir", default=default_input_dir, type=Path)
    parser.add_argument("--output_dir", default=default_output_data_dir, type=Path)
    parser.add_argument("--input_data_dir", default=default_input_data_dir, type=Path)

    args = vars(parser.parse_args())
    return args


def read_scores(score_file: Path) -> dict:
    if not score_file.exists():
        return {}
    with open(score_file, "r", encoding="utf-8") as score_file_obj:
        print(score_file)
        scores = json.load(score_file_obj)
        return scores


def write_scores(score_file: Path, scores: dict):
    with open(score_file, "w", encoding="utf-8") as score_file_obj:
        json.dump(scores, score_file_obj)


def get_duration(prediction_metadata_file):
    with open(
        prediction_metadata_file, "r", encoding="utf-8"
    ) as prediction_metadata_file_obj:
        metadata = json.load(prediction_metadata_file_obj)
        return metadata["ingestion_duration"]


def main():
    args = parse_args()

    output_dir: Path = args["output_dir"]
    output_dir.mkdir(exist_ok=True)

    # reference_dir = os.path.join(args["input_dir"], "ref")  # Ground truth data
    input_data_dir = args["input_data_dir"]
    prediction_dir = args["input_dir"]
    prediction_file = prediction_dir / "prediction"
    score_file = output_dir / "scores.json"
    # html_file = args["output_dir"] / "detailed_results.html"

    dataset = AutoMLCupDataset(input_data_dir)
    y_test = np.array(dataset.get_split("test")["label"])

    y_pred = np.genfromtxt(prediction_file, skip_header=1)

    accuracy = accuracy_score(y_test, y_pred)

    scores = read_scores(score_file)
    scores[dataset.name()] = accuracy
    scores["duration"] = get_duration(prediction_dir / "end.txt")
    write_scores(score_file, scores)


if __name__ == "__main__":
    main()
