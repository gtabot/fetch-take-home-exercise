import pytorch_lightning as pl
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class SentenceTransformer(pl.LightningModule):
    def __init__(self, model_name: str = MODEL_NAME) -> None:
        super().__init__()
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.embedding_length = self.model.config.hidden_size

    def forward(self, text: str | list[str]) -> torch.Tensor:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        # Move input tensors to the same device as the model
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        sentence_embedding = outputs.last_hidden_state.mean(dim=1)
        return sentence_embedding
