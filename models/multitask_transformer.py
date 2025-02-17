import torch
import torch.nn as nn

from data.multitask_datamodule import MultitaskDataModule
from models.sentence_transformer import SentenceTransformer

HIDDEN_DIM = 256


class MultitaskTransformer(SentenceTransformer):
    def __init__(
        self,
        datamodule: MultitaskDataModule,
        hidden_dim: int = HIDDEN_DIM,
        learning_rate: float = 1e-4,
    ) -> None:
        super().__init__()
        self.datamodule = datamodule
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        # Freeze the sentence transformer weights
        for param in self.sentence_transformer.parameters():
            param.requires_grad = False
        self.category_classifier = nn.Sequential(
            nn.Linear(self.embedding_length, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, datamodule.category_dataset.num_classes),
        )
        self.sentiment_classifier = nn.Sequential(
            nn.Linear(self.embedding_length, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, datamodule.sentiment_dataset.num_classes),
        )

    def forward(
        self, text: str | list[str]
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        embedding = super().forward(text)
        return {
            "embedding": embedding,
            "category_logits": self.category_classifier(embedding),
            "sentiment_logits": self.sentiment_classifier(embedding),
        }

    def training_step(self, batch: dict[str, list[str] | torch.Tensor]) -> torch.Tensor:
        category_batch, sentiment_batch = batch

        embeddings = super().forward(category_batch["text"])
        category_logits = self.category_classifier(embeddings)
        category_loss = self.loss_fn(category_logits, category_batch["label"])
        self.log("train_category_loss", category_loss)

        embeddings = super().forward(sentiment_batch["text"])
        sentiment_logits = self.sentiment_classifier(embeddings)
        sentiment_loss = self.loss_fn(sentiment_logits, sentiment_batch["label"])
        self.log("train_sentiment_loss", sentiment_loss)

        combined_loss = (category_loss + sentiment_loss) / 2
        self.log("train_combined_loss", combined_loss)
        return combined_loss

    def validation_step(self, batch: dict[str, list[str] | torch.Tensor]) -> torch.Tensor:
        category_batch, sentiment_batch = batch

        try:
            embeddings = super().forward(category_batch["text"])
            category_logits = self.category_classifier(embeddings)
            category_loss = self.loss_fn(category_logits, category_batch["label"])
            self.log("val_category_loss", category_loss)
                
            embeddings = super().forward(sentiment_batch["text"])
            sentiment_logits = self.sentiment_classifier(embeddings)
            sentiment_loss = self.loss_fn(sentiment_logits, sentiment_batch["label"])
            self.log("val_sentiment_loss", sentiment_loss)
            
            combined_loss = (category_loss + sentiment_loss) / 2
            self.log("val_combined_loss", combined_loss)
            return combined_loss
        except Exception as e:
            pass

    def predict_step(self, text: str | list[str]) -> dict[str, torch.Tensor | str]:
        output = self.forward(text)
        category_pred = torch.argmax(output["category_logits"], dim=1)
        category_label = self.datamodule.category_dataset.idx_to_label[
            category_pred.item()
        ]
        sentiment_pred = torch.argmax(output["sentiment_logits"], dim=1)
        sentiment_label = self.datamodule.sentiment_dataset.idx_to_label[
            sentiment_pred.item()
        ]
        return {
            "embedding": output["embedding"],
            "category": category_label,
            "sentiment": sentiment_label,
        }

    def loss_fn(self, logits: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(logits, label)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
