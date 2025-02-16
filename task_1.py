import torch

from models.sentence_transformer import SentenceTransformer


def load_sentences(file_path: str) -> list[str]:
    with open(file_path, "r") as file:
        return [line.strip() for line in file.readlines()]


def main():
    sentence_transformer = SentenceTransformer()
    sentence_transformer.eval()
    sentences = load_sentences("data/sentences.txt")
    with torch.no_grad():
        for sentence in sentences:
            embedding = sentence_transformer(sentence)
            print(f"Sentence: {sentence}")
            print(f"Embedding: {embedding}")
            print(f"\n{'-' * 80}\n")  # Separator


if __name__ == "__main__":
    main()
