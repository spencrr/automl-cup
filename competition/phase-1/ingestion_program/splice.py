from pathlib import Path
from dataloader import AutoMLCupDataloader
from datasets import load_dataset, DatasetDict


class SpliceDataloader(AutoMLCupDataloader):
    @staticmethod
    def name():
        return "splice"

    test_size = 0.1
    shuffle = False

    def __init__(self, directory: Path, **kwargs):
        super().__init__(directory, **kwargs)

        cache_directory = directory / "cache"
        if cache_directory.exists():
            self.dataset = DatasetDict.load_from_disk(cache_directory)
        else:

            def positions_to_vec(example):
                example["sequence"] = list(example[f"position_{i}"] for i in range(60))
                return example

            self.dataset = (
                load_dataset(
                    "mstz/splice",
                    "splice",
                    data_dir=directory / SpliceDataloader.name(),
                )["train"]
                .map(
                    positions_to_vec,
                    remove_columns=list(f"position_{i}" for i in range(60)),
                )
                .train_test_split(
                    test_size=SpliceDataloader.test_size,
                    shuffle=SpliceDataloader.shuffle,
                )
            )

            self.dataset.save_to_disk(cache_directory)

    def get_train(self):
        return self.dataset["train"]

    def get_val(self):
        return self.dataset["test"]
