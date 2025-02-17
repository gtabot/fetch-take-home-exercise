import multiprocessing
import pytorch_lightning as pl
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, RandomSampler
from typing import Optional

CATEGORY_DATASET_NAME = "keelezibel/sentence_classification_dataset"
SENTIMENT_DATASET_NAME = "Sp1786/multiclass-sentiment-analysis-dataset"


class CategoryDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[str]):
        self.texts = texts
        self.num_classes = len(set(labels))
        self.label_to_idx = {
            label: idx for idx, label in enumerate(sorted(set(labels)))
        }
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        self.labels = torch.tensor([self.label_to_idx[label] for label in labels])
        self.dataset_name = CATEGORY_DATASET_NAME

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, str | torch.Tensor]:
        return {
            "task": "category",
            "text": self.texts[idx],
            "label": self.labels[idx],
        }


class SentimentDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int]):
        self.texts = texts
        self.num_classes = len(set(labels))
        self.idx_to_label = {0: "negative", 1: "neutral", 2: "positive"}
        self.labels = labels
        self.dataset_name = SENTIMENT_DATASET_NAME

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, str | torch.Tensor]:
        return {
            "task": "sentiment",
            "text": self.texts[idx],
            "label": self.labels[idx],
        }


class MultitaskDataModule(pl.LightningDataModule):
    category_dataset: CategoryDataset
    sentiment_dataset: SentimentDataset

    def __init__(self, batch_size: int = 32) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.cuda_is_available = torch.cuda.is_available()
        self.num_workers = max(multiprocessing.cpu_count() - 1, 1)

    def setup(self, stage: Optional[str] = None) -> None:
        category_data = load_dataset(CATEGORY_DATASET_NAME)
        self.category_dataset = CategoryDataset(
            category_data["train"]["text"], category_data["train"]["label"]
        )
        self.category_val_dataset = CategoryDataset(
            category_data["test"]["text"], category_data["test"]["label"]
        )
        sentiment_data = load_dataset(SENTIMENT_DATASET_NAME)
        self.sentiment_dataset = SentimentDataset(
            sentiment_data["train"]["text"], sentiment_data["train"]["label"]
        )
        self.sentiment_val_dataset = SentimentDataset(
            sentiment_data["test"]["text"], sentiment_data["test"]["label"]
        )

    def train_dataloader(self) -> tuple[DataLoader, DataLoader]:
        # Create random samplers to balance the dataset sizes
        if len(self.category_dataset) > len(self.sentiment_dataset):
            category_sampler = None
            sentiment_sampler = RandomSampler(
                self.sentiment_dataset,
                replacement=True,
                num_samples=len(self.category_dataset),
            )
        else:
            sentiment_sampler = None
            category_sampler = RandomSampler(
                self.category_dataset,
                replacement=True,
                num_samples=len(self.sentiment_dataset),
            )
        # Create dataloaders
        category_dataloader = DataLoader(
            self.category_dataset,
            batch_size=self.batch_size,
            sampler=category_sampler,
            num_workers=self.num_workers,
            pin_memory=self.cuda_is_available,
            persistent_workers=True,
        )
        sentiment_dataloader = DataLoader(
            self.sentiment_dataset,
            batch_size=self.batch_size,
            sampler=sentiment_sampler,
            num_workers=self.num_workers,
            pin_memory=self.cuda_is_available,
            persistent_workers=True,
        )
        return category_dataloader, sentiment_dataloader

    def val_dataloader(self) -> tuple[DataLoader, DataLoader]:
        category_val_loader = DataLoader(
            self.category_val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.cuda_is_available,
            persistent_workers=True,
        )
        sentiment_val_loader = DataLoader(
            self.sentiment_val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.cuda_is_available,
            persistent_workers=True,
        )
        return category_val_loader, sentiment_val_loader
