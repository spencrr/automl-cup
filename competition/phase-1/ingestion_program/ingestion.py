# pylint: disable=logging-fstring-interpolation, broad-except
"""ingestion program for autoWSL"""
import argparse
import datetime
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from common import Timer, get_logger
from dataset import AutoMLCupDataset
from filelock import FileLock
from predict import predict

VERBOSITY_LEVEL = "INFO"
LOGGER = get_logger(VERBOSITY_LEVEL, __file__)


def _cwd() -> Path:
    """Helper function for getting the current directory of this script."""
    return Path(__file__).resolve().parent


def write_start_file(output_dir):
    """Create start file 'start.txt' in `output_dir` with updated timestamp
    start time.

    """
    LOGGER.info("===== alive_thd started")
    start_filepath = output_dir / "start.txt"
    lockfile = output_dir / "start.txt.lock"
    while True:
        current_time = datetime.datetime.now().timestamp()
        with FileLock(lockfile):
            with open(start_filepath, "w", encoding="utf-8") as ftmp:
                json.dump(current_time, ftmp)
        time.sleep(10)


class ModelApiError(Exception):
    """Model api error"""


class IngestionError(RuntimeError):
    """Model api error"""


def _parse_args():
    root_dir = _cwd()
    default_dataset_dir = root_dir / "sample_data"
    default_output_dir = root_dir / "sample_result_submission"
    default_ingestion_program_dir = root_dir / "ingestion_program"
    default_code_dir = root_dir / "code_submission"
    default_score_dir = root_dir / "scoring_output"
    default_temp_dir = root_dir / "temp_output"
    default_time_budget = 1200
    default_pred_time_budget = 600
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=default_dataset_dir,
        help="Directory storing the dataset (containing " "e.g. adult.data/)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=default_output_dir,
        help="Directory storing the predictions. It will "
        "contain e.g. [start.txt, adult.predict_0, "
        "adult.predict_1, ..., end.txt] when ingestion "
        "terminates.",
    )
    parser.add_argument(
        "--ingestion_program_dir",
        type=Path,
        default=default_ingestion_program_dir,
        help="Directory storing the ingestion program "
        "`ingestion.py` and other necessary packages.",
    )
    parser.add_argument(
        "--code_dir",
        type=Path,
        default=default_code_dir,
        help="Directory storing the submission code "
        "`model.py` and other necessary packages.",
    )
    parser.add_argument(
        "--score_dir",
        type=Path,
        default=default_score_dir,
        help="Directory storing the scoring output "
        "e.g. `scores.txt` and `detailed_results.html`.",
    )
    parser.add_argument(
        "--temp_dir",
        type=Path,
        default=default_temp_dir,
        help="Directory storing the temporary output."
        "e.g. save the participants` model after "
        "trainning.",
    )
    parser.add_argument(
        "--time_budget",
        type=float,
        default=default_time_budget,
        help="Time budget for trainning model if not specified" " in meta.json.",
    )
    parser.add_argument(
        "--pred_time_budget",
        type=float,
        default=default_pred_time_budget,
        help="Time budget for predicting model " "if not specified in meta.json.",
    )
    args = parser.parse_args()
    LOGGER.debug(f"Parsed args are: {args}")
    LOGGER.debug("-" * 50)
    if args.dataset_dir.endswith("run/input") and args.code_dir.endswith("run/program"):
        LOGGER.debug(
            "Since dataset_dir ends with 'run/input' and code_dir "
            "ends with 'run/program', suppose running on "
            "CodaLab platform. Modify dataset_dir to 'run/input_data'"
            " and code_dir to 'run/submission'. "
            "Directory parsing should be more flexible in the code of"
            " compute worker: we need explicit directories for "
            "dataset_dir and code_dir."
        )

        args.dataset_dir = args.dataset_dir.replace("run/input", "run/input_data")
        args.code_dir = args.code_dir.replace("run/program", "run/submission")

        # Show directories for debugging
        LOGGER.debug(f"sys.argv = {sys.argv}")
        LOGGER.debug(f"Using dataset_dir: {args.dataset_dir}")
        LOGGER.debug(f"Using output_dir: {args.output_dir}")
        LOGGER.debug(f"Using ingestion_program_dir: {args.ingestion_program_dir}")
        LOGGER.debug(f"Using code_dir: {args.code_dir}")
    return args


def _init_python_path(args):
    sys.path.append(args.ingestion_program_dir)
    sys.path.append(args.code_dir)
    args.output_dir.mkdir(exist_ok=True)
    args.temp_dir.mkdir(exist_ok=True)


def _check_umodel_methed(umodel):
    # Check if the model has methods `train`, `test`.
    for attr in ["train", "test"]:
        if not hasattr(umodel, attr):
            raise ModelApiError(
                "Your model object doesn't have the method "
                f"`{attr}`. Please implement it in model.py."
            )


def _train(args, umodel, dataset):
    # Train the model
    timer = Timer()
    timer.set(args.time_budget)
    train_dataset, train_label = dataset.get_train()
    with timer.time_limit("training"):
        umodel.train(train_dataset, train_label)
    duration = timer.duration
    LOGGER.info(f"Finished training the model. time spent {duration:5.2} sec")

    result = {}
    result["duration"] = duration
    return result


def _predict(args):
    # Make predictions using the trained model
    LOGGER.info("===== call prediction")

    predict_args = {
        "dataset_dir": args.dataset_dir,
        "model_dir": args.code_dir,
        "output_dir": args.output_dir,
        "temp_dir": args.temp_dir,
        "pred_time_budget": args.pred_time_budget,
    }
    result = predict(predict_args)
    return result


def _finalize(args, train_result, pred_result):
    if pred_result["status"] == "success":
        # Finishing ingestion program
        end_time = time.time()
        overall_time_spent = train_result["duration"] + pred_result["duration"]

        # Write overall_time_spent to a end.txt file
        end_filename = "end.txt"
        content = {"ingestion_duration": overall_time_spent, "end_time": end_time}

        with open(args.output_dir / end_filename, "w", encoding="utf-8") as ftmp:
            json.dump(content, ftmp)
            LOGGER.info(f"Wrote the file {end_filename} marking the end of ingestion.")

            LOGGER.info("[+] Done. Ingestion program successfully terminated.")
            LOGGER.info(f"[+] Overall time spent {overall_time_spent:5.2} sec")

        # Copy all files in output_dir to score_dir
        os.system(f"cp -R {args.output_dir/ '*'} {args.score_dir}")
        LOGGER.debug("Copied all ingestion output to scoring output directory.")

        LOGGER.info("[Ingestion terminated]")
    elif pred_result["status"] == "timeout":
        raise IngestionError("predicting timeout")
    else:
        raise IngestionError("error occurs when predicting")


def main():
    """main entry"""
    LOGGER.info("===== Start ingestion program.")
    # Parse directories from input arguments
    LOGGER.info("===== Initialize args.")
    args = _parse_args()
    _init_python_path(args)
    dataset = AutoMLCupDataset(args.dataset_dir)
    LOGGER.info("===== Load metadata.")
    metadata = dataset.get_metadata()
    args.time_budget = metadata.get("time_budget", args.time_budget)
    args.pred_time_budget = metadata.get("pred_time_budget", args.pred_time_budget)
    LOGGER.info(f"Time budget: {args.time_budget}")

    LOGGER.info("===== Set alive_thd")
    alive_thd = threading.Thread(
        target=write_start_file, name="alive", args=(args.output_dir,)
    )
    alive_thd.daemon = True
    alive_thd.start()

    LOGGER.info("===== Install user dependencies")
    requirements_txt = args.ingestion_program_dir / "requirements.txt"
    if requirements_txt.exists():
        subprocess.check_call([sys.executable, "-m", "pip", "-r", requirements_txt])

    LOGGER.info("===== Load user model")
    # pylint: disable-next=import-error,import-outside-toplevel
    from model import Model

    umodel = Model(metadata)

    LOGGER.info("===== Check user model methods")
    _check_umodel_methed(umodel)

    LOGGER.info("===== Begin training user model")
    train_result = _train(args, umodel, dataset)

    LOGGER.info("===== Begin preding by user model on test set")
    pred_result = _predict(args)

    _finalize(args, train_result, pred_result)


if __name__ == "__main__":
    main()
