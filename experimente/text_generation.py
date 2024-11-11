# import standard modules
import os

# third party modules
import torch
import mlflow
import lightning as pl
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tokenizers import Tokenizer

from omegaconf import OmegaConf
from dacite import from_dict
from torchinfo import summary
from dacite import Config as DaciteConfig
from xlstm.xlstm import xLSTMLMModel, xLSTMLMModelConfig
from torch.utils.data import DataLoader

# project related modules
from data.potter import HarryPotterDataset

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f'DEVICE: {device}')
# Datensatz aufrufen
file_path = os.path.join(os.getcwd(), "data/harrypotter_one.txt")
tokenizer_path = os.path.join(os.getcwd(), "data/bpe_tokenizer.json")
seq_length = 100
dataset = HarryPotterDataset(file_path, seq_length, device, tokenizer_path)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


# create torch lightning model
class xLSTMBaselineLanguageGen(pl.LightningModule):

    def __init__(self, cfg, lr: float = 1e-2):
        super().__init__()  # Add the super().__init__() call
        self.lr = lr
        self.tokenizer = Tokenizer.from_file(
            os.path.join(os.getcwd(), "data/bpe_tokenizer.json")
        )
        self.xlstm = xLSTMLMModel(cfg)
        

    def on_train_start(self):
        pass

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()
        X, y = batch
        logits = self.xlstm(X) # out)

        loss = F.cross_entropy(logits[:, -1, :], y.to(torch.float))
        mlflow.log_metric("train_loss", loss.item())

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


xlstm_cfg = f""" 
vocab_size: {dataset.tokenizer.get_vocab_size()}
mlstm_block:
  mlstm:
    conv1d_kernel_size: 4
    qkv_proj_blocksize: 4
    num_heads: 4
context_length: {seq_length}
num_blocks: 1
embedding_dim: 512
"""

max_epochs = 100

mlflow.set_tracking_uri("http://localhost:8080")
mlflow.set_experiment("text_gen_potter")
with mlflow.start_run(run_name="baseline[mlstm:1|slstm:0]"):
    mlflow.pytorch.autolog()

    cfg_json = OmegaConf.create(xlstm_cfg)
    cfg = from_dict(
        data_class=xLSTMLMModelConfig,
        data=OmegaConf.to_container(cfg_json),
        config=DaciteConfig(strict=True),
    )

    # log the entire configuration for training
    for key, value in cfg_json.items():
        mlflow.log_param(key, value)

    model = xLSTMBaselineLanguageGen(cfg).to(device)

    # Log model summary.
    with open("model_summary.txt", "w") as f:
        f.write(str(summary(model)))
    mlflow.log_artifact("model_summary.txt")

    # Configure trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator='auto',  # Automatically detect if GPU is available
        devices=[0],
        # logger=mlf_logger,
        enable_progress_bar=True,
    )

    # Train the model
    trainer.fit(model, dataloader)
