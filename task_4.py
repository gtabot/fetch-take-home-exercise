import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from data.multitask_datamodule import MultitaskDataModule
from models.multitask_transformer import MultitaskTransformer


# will trade-off precision for performance
torch.set_float32_matmul_precision("high")

CHECKPOINTS_DIR = "checkpoints"


def main():
    datamodule = MultitaskDataModule()
    datamodule.setup()
    model = MultitaskTransformer(datamodule)
    trainer = pl.Trainer(
        max_epochs=-1,
        logger=pl.loggers.TensorBoardLogger(save_dir="logs", name=""),
        callbacks=[
            # Early stopping based on combined loss
            EarlyStopping(monitor="val_combined_loss", patience=2),
            # Save best checkpoint for category task
            ModelCheckpoint(
                dirpath=CHECKPOINTS_DIR,
                filename="best-category-loss_{epoch:02d}_{val_category_loss:.3f}",
                monitor="val_category_loss",
            ),
            # Save best checkpoint for sentiment task
            ModelCheckpoint(
                dirpath=CHECKPOINTS_DIR,
                filename="best-sentiment-loss_{epoch:02d}_{val_sentiment_loss:.3f}",
                monitor="val_sentiment_loss",
            ),
        ],
    )
    trainer.fit(model, datamodule)
    trainer.save_checkpoint(f"{CHECKPOINTS_DIR}/final-multitask-transformer.ckpt")


if __name__ == "__main__":
    main()
