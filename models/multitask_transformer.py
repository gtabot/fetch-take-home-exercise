import torch
import torch.nn as nn

from models.sentence_transformer import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_LENGTH = 128
HIDDEN_DIM = 256
CATEGORIES = ["question", "declarative", "imperative"]
SENTIMENTS = ["negative", "neutral", "positive"]


# Inherit from SentenceTransformer to use sentence embeddings forward method
class MultitaskTransformer(SentenceTransformer):
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        embedding_length: int = EMBEDDING_LENGTH,
        hidden_dim: int = HIDDEN_DIM,
        categories: list[str] = CATEGORIES,
        sentiments: list[str] = SENTIMENTS,
        learning_rate: float = 1e-4,
    ) -> None:
        super().__init__(model_name, embedding_length)
        self.model_name = model_name
        self.embedding_length = embedding_length
        self.hidden_dim = hidden_dim
        self.categories = categories
        self.sentiments = sentiments
        self.learning_rate = learning_rate
        # Create forward pass classifier for each task
        self.category_classifier = nn.Sequential(
            nn.Linear(embedding_length, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(categories)),
        )
        self.sentiment_classifier = nn.Sequential(
            nn.Linear(embedding_length, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(sentiments)),
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

    def loss_fn(self, logits: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(logits, label)

    def training_step(
        self, batch: dict[str, list[str] | torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        task = batch["task"][0]
        outputs = self.forward(batch["text"])
        if task == "category":
            loss = self.loss_fn(outputs["category_logits"], batch["label"])
            self.log("train_category_loss", loss)
        elif task == "sentiment":
            loss = self.loss_fn(outputs["sentiment_logits"], batch["label"])
            self.log("train_sentiment_loss", loss)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
