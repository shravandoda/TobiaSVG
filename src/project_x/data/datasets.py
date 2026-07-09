"""Load base and repair SVG datasets from local disk or the Hugging Face Hub."""

from logging import getLogger

from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_dataset
from torch.utils.data import Dataset as TorchDataset

from src.project_x.utils.logger import setup_logging

setup_logging()
logger = getLogger(__name__)


# Config passed here will come later. The config will help sample
def get_tobias_dataset() -> DatasetDict:
    logger.info("----------------- Loading Tobias Dataset ----------------- ")
    tobias = load_dataset("shravandoda/TobiaSVG")
    logger.info("Dataset splits: %s", len(tobias))
    logger.info("Column names: %s", tobias.column_names)
    logger.info("---------------------------------------------------------- ")
    return tobias


# Config passed here will come later. The config will help sample
def get_tobias_repair_dataset() -> DatasetDict:
    logger.info("----------------- Loading Tobias Dataset ----------------- ")
    tobias_repair = load_dataset("shravandoda/TobiaSVG-repair")
    logger.info("Dataset splits: %s", len(tobias_repair))
    logger.info("Column names: %s", tobias_repair.column_names)
    logger.info("---------------------------------------------------------- ")
    return tobias_repair


class TextToSVGDataset(TorchDataset):
    def __init__(self, dataset: HFDataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        row = self.dataset[index]
        text = row["text"]
        svg = row["svg"]

        return {"text": text, "svg": svg}


class ImageToSVGDataset(TorchDataset):
    def __init__(self, dataset: HFDataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        row = self.dataset[index]
        svg = row["svg"]

        return {"svg": svg}


class SVGRepairDataset(TorchDataset):
    def __init__(self, dataset: HFDataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        row = self.dataset[index]
        svg = row["svg"]
        corrupted_svg = row["corrupted_svg"]

        return {"svg": svg, "corrupted_svg": corrupted_svg}
