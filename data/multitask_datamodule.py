import pytorch_lightning as pl
import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from typing import Optional

CATEGORY_DATASET_NAME = "keelezibel/sentence_classification_dataset"
SENTIMENT_DATASET_NAME = "Sp1786/multiclass-sentiment-analysis-dataset"


class CategoryDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int]):
        self.texts = texts
        self.labels = labels
        self.dataset_name = CATEGORY_DATASET_NAME

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        return {
            "task": "category",
            "text": self.texts[idx],
            "label": torch.tensor(self.labels[idx]),
        }


class SentimentDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int]):
        self.texts = texts
        self.labels = labels
        self.dataset_name = SENTIMENT_DATASET_NAME

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        return {
            "task": "sentiment",
            "text": self.texts[idx],
            "label": torch.tensor(self.labels[idx]),
        }


class MultitaskDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 32) -> None:
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None) -> None:
        category_data = load_dataset(CATEGORY_DATASET_NAME)
        self.category_dataset = CategoryDataset(
            category_data["train"]["text"], category_data["train"]["label"]
        )
        sentiment_data = load_dataset(SENTIMENT_DATASET_NAME)
        self.sentiment_dataset = SentimentDataset(
            sentiment_data["train"]["text"], sentiment_data["train"]["label"]
        )

    def train_dataloader(self) -> tuple[DataLoader, DataLoader]:
        category_dataloader = DataLoader(
            self.category_dataset, batch_size=self.batch_size, shuffle=True
        )
        sentiment_dataloader = DataLoader(
            self.sentiment_dataset, batch_size=self.batch_size, shuffle=True
        )
        return category_dataloader, sentiment_dataloader
