"""AutoML datasets."""

from pathlib import Path
from typing import List, Optional, Type, TypeVar

from splice import SpliceDataloader

from common import get_logger
from dataloader import AutoMLCupDataloader
from datasets import Dataset
from listops import ListOpsDataloader

VERBOSITY_LEVEL = "WARNING"
LOGGER = get_logger(VERBOSITY_LEVEL, __file__)


class AutoMLCupDataset:
    """AutoMLCupDataset"""

    D = TypeVar("D", bound=AutoMLCupDataloader)
    dataloaders: List[Type[D]] = [ListOpsDataloader, SpliceDataloader]

    def __init__(self, directory: Path):
        """init"""
        dataset: Optional[AutoMLCupDataloader] = None
        dataset_name = directory.name

        for dataloader in AutoMLCupDataset.dataloaders:
            if dataset_name == dataloader.name():
                dataset = dataloader(directory)
                break

        if dataset is None:
            raise ValueError(f"Dataset from {directory} not found.")
        self.dataset = dataset

    def get_train(self) -> Dataset:
        return self.dataset.get_train()

    def get_val(self) -> Dataset:
        return self.dataset.get_val()
