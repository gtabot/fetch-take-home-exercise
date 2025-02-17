# Machine Learning Engineer Take Home Exercise

This project builds a multi-task learning model capable of classifying both sentence categories (e.g., "declarative," "imperative," "question") and sentiment (e.g., "negative," "neutral," "positive") from text data.  

I leverage [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) to streamline training configuration, GPU device management, and seamless integration with Hugging Face's libraries.

## Setup

It is recommended to use a virtual environment to install the dependencies.

```bash
python -m venv .venv
source .venv/bin/activate # on macOS/Linux
.venv\Scripts\activate    # on Windows
```

Install the dependencies.

```bash
poetry install
```

## Task 1: Sentence Transformer Implementation

The file `models/sentence_transformer.py` contains the implementation of the Sentence Transformer model.

- I use a pre-trained Sentence Transformer model and text tokenizer from HuggingFace: [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
  - The model is lightweight, fast, and specialized for sentence-level embeddings
  - It also has good performance despite its small size

- The model encodes text into an embedding of fixed size (384)
  - All text sequences are either padded or truncated to the same embedding size
  - The embedding size is determined by the pre-trained model architecture and is computed as the mean of the last hidden layer of the model

### Testing the Sentence Transformer Model

The file `task_1.py` contains the code to test the Sentence Transformer model.

- The model is instantiated and set to evaluation mode
- The sentences are loaded from the `data/sentences.txt` file
- The model is used to encode the sentences
- Both the sentences and their embeddings are printed to the console

```bash
$ python task_1.py

Sentence: I absolutely love this new restaurant, the food is amazing!
Embedding: tensor([[-1.6839e-01, -3.9668e-01,  4.2607e-01,  1.9269e-01, ...]
```

Full output: [task_1.txt](task_1.txt)

## Task 2: Multi-Task Learning Model Implementation

The file `models/multitask_transformer.py` contains the implementation of the multi-task learning model.

- The model inherits from the `SentenceTransformer` class in order to leverage the pre-trained model and text tokenizer
- The `SentenceTransformer` outputs an embedding which is then passed to two separate classifier heads:
  - One for the category classification task
  - One for the sentiment classification task
- The model uses a simple feedforward network for each head
- The model uses a cross-entropy loss function for each head and averages the two losses to derive a single combined loss
- During forward pass, the model outputs a dictionary containing the embeddings and the logits for each head

### Task A: Sentence Classification

I use the HuggingFace dataset [keelezibel/sentence_classification_dataset](https://huggingface.co/datasets/keelezibel/sentence_classification_dataset) for the category classification task. The dataset contains 31k training sentences with their corresponding category labels: "*declarative*", "*imperative*", and "*question*".

### Task B: Sentiment Classification

I use the HuggingFace dataset [Sp1786/multiclass-sentiment-analysis-dataset](https://huggingface.co/datasets/Sp1786/multiclass-sentiment-analysis-dataset) for the sentiment classification task. The dataset contains 31.2k training sentences with their corresponding sentiment labels: "*negative*", "*neutral*", and "*positive*".

## Task 3: Training Considerations

**Discuss the implications and advantages of each scenario and explain your rationale as to how the model should be trained given the following:**

1. If the entire network should be frozen.

- Network is not very malleable to new tasks or data
- Useful when data to train is similar to pre-trained data

2. If only the transformer backbone should be frozen.

- The transformer is used as a feature extractor and is not trained
- The task-specific heads only are trained
- Good when pre-trained embeddings are useful for classification

3. If only one of the task-specifc heads should be frozen.

- Allows one class to be trained while keeping the other stable
- Can be useful when one dataset was introduced later in time and is much smaller

**Consider a scenario where transfer learning can be beneficial. Explain how you would approach the transfer learning process, including:**

1. The choice of a pre-trained model.

- Choose a popular pre-trained model that embeds text sentences into a fixed-size vector
- The pre-trained model should be lightweight and fast

2. The layers you would freeze/unfreeze.

- Freeze the transformer backbone to keep the pre-trained embeddings stable
- Unfreeze the task-specific heads to train the model from the pre-trained embeddings

3. The rationale behind these choices.

- This takes advantage of transfer learning by leveraging a pre-trained model that is trained on a large corpus of text
- This also prevents the model from overfitting to the new task and allows it to generalize better to all text
- The task-specific heads are fine-tuned on the new tasks

## Task 4: Training Loop Implementation (BONUS)

The file `task_4.py` contains the implementation of the training loop.

- The datasets are loaded and prepared for training in `MultitaskDataModule`
  - Inequal dataset sizes are handled with random sampling to lengthen the shorter dataset
- The datasets are used to instantiate the multi-task learning model and determine the number of classes for each task
- The training loop is implemented using PyTorch Lightning
  - Training will continue until the early stopping callback is triggered (no loss improvement)
  - The final model and other checkpoints are saved in the `checkpoints` directory (ignored by `.gitignore`)
    - `final-multitask-transformer.ckpt`
    - `best-category-loss_epoch={##}_loss={#.###}.ckpt`
    - `best-sentiment-loss_epoch={##}_loss={#.###}.ckpt`
  - TensorBoard logs are saved in the `logs` directory

### Training the Multi-Task Learning Model

```bash
$ python task_4.py

GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
HPU available: False, using: 0 HPUs
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]

  | Name                 | Type       | Params | Mode
------------------------------------------------------------
0 | model                | BertModel  | 22.7 M | eval
1 | category_classifier  | Sequential | 99.3 K | train
2 | sentiment_classifier | Sequential | 99.3 K | train
------------------------------------------------------------
22.9 M    Trainable params
0         Non-trainable params
22.9 M    Total params
91.648    Total estimated model params size (MB)
8         Modules in train mode
120       Modules in eval mode
...
```

## Task 5: Evaluation (BONUS)

The file `task_5.py` contains evaluation for the multi-task learning model.

- The model is loaded using `checkpoints/final-multitask-transformer.ckpt`
- The sentences are loaded from `data/sentences.txt`
- The model is used to predict the category and sentiment for each sentence
- The results are printed to the console

```bash
$ python task_5.py

Sentence: I absolutely love this new restaurant, the food is amazing!
Category: imperative
Sentiment: positive
...
```

Full output: [task_5.txt](task_5.txt)
