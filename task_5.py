import torch

from data.multitask_datamodule import MultitaskDataModule
from models.multitask_transformer import MultitaskTransformer


def load_sentences(file_path: str) -> list[str]:
    with open(file_path, "r") as file:
        return [line.strip() for line in file.readlines()]


def main():
    datamodule = MultitaskDataModule()
    datamodule.setup()
    model = MultitaskTransformer.load_from_checkpoint(
        checkpoint_path="checkpoints/final-multitask-transformer.ckpt",
        datamodule=datamodule,
    )
    model.eval()
    sentences = load_sentences("data/sentences.txt")
    with torch.no_grad():
        for sentence in sentences:
            predictions = model.predict_step(sentence)
            print(f"\nSentence: {sentence}")
            print(f"Category: {predictions['category']}")
            print(f"Sentiment: {predictions['sentiment']}")
            print("-" * 80)


if __name__ == "__main__":
    main()
