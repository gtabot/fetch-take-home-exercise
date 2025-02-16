import pytorch_lightning as pl
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_LENGTH = 128


class SentenceTransformer(pl.LightningModule):
    def __init__(
        self, model_name: str = MODEL_NAME, embedding_length: int = EMBEDDING_LENGTH
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.embedding_length = embedding_length
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def forward(self, text: str | list[str]) -> torch.Tensor:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.embedding_length,
            padding="max_length",
            truncation=True,
        )
        outputs = self.model(**inputs)
        sentence_embedding = outputs.last_hidden_state.mean(dim=1)
        return sentence_embedding
