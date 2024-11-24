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
from mlflow import MlflowClient
from omegaconf import OmegaConf
from dacite import from_dict
from torchinfo import summary
from dacite import Config as DaciteConfig
from xlstm.xlstm import xLSTMLMModel, xLSTMLMModelConfig
from torch.utils.data import DataLoader
from xlstm.experiments.lr_scheduler import LinearWarmupCosineAnnealing
from dataclasses import dataclass
from lightning.pytorch.loggers import MLFlowLogger

mlf_logger = MLFlowLogger(experiment_name="lightning_logs", tracking_uri="file:./ml-runs", log_model=True)

# project related modules
from data.potter import HarryPotterDataset

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# hyper parameters
seq_length = 200
max_epochs = 150

# Datensatz aufrufen
file_path = os.path.join(os.getcwd(), "experimente/data/harrypotter_one.txt")
tokenizer_path = os.path.join(os.getcwd(), "experimente/data/bpe_tokenizer.json")
dataset = HarryPotterDataset(file_path, seq_length, device, tokenizer_path)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

torch_dtype_map: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}

@dataclass
class TrainingConfig:
    batch_size: int
    lr: float
    seed: int
    val_every_step: int
    lr_warmup_steps: int
    lr_decay_until_steps: int
    lr_decay_factor: float
    weight_decay: float
    num_steps: int
    device: str
    amp_precision: str
    weight_precision: str
    enable_mixed_precision: bool

def print_auto_logged_info(r):
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
    print(f"run_id: {r.info.run_id}")
    print(f"artifacts: {artifacts}")
    print(f"params: {r.data.params}")
    print(f"metrics: {r.data.metrics}")
    print(f"tags: {tags}")

# create torch lightning model
class xLSTMLitModel(pl.LightningModule):
    def __init__(self, model_config, training_cfg):
        super().__init__()
        self.model = xLSTMLMModel(model_config)
        self.model.reset_parameters()
        self.model = self.model.to(dtype=torch_dtype_map[training_cfg.weight_precision])
        self.model_config = model_config
        self.training_cfg = training_cfg

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)[:, -1, :]
        loss = nn.functional.cross_entropy(
            outputs.view(-1, self.model_config.vocab_size),
            labels.view(-1),
            ignore_index=-1,
        )
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optim_groups = self.model._create_weight_decay_optim_groups()
        optimizer = optim.AdamW(
            [
                {"weight_decay": self.training_cfg.weight_decay, "params": optim_groups[0]},
                {"weight_decay": 0.0, "params": optim_groups[1]},
            ],
            lr=self.training_cfg.lr,
        )
        scheduler = {
            'scheduler': LinearWarmupCosineAnnealing(
                optimizer,
                self.training_cfg.lr_warmup_steps,
                self.training_cfg.lr_decay_until_steps,
                self.training_cfg.lr,
                self.training_cfg.lr_decay_factor * self.training_cfg.lr,
            ),
            'interval': 'step',
            'frequency': 1,
        }
        return [optimizer], [scheduler]



xlstm_cfg = f"""
model:
  num_blocks: 1
  embedding_dim: 512
  mlstm_block:
    mlstm:
      num_heads: 1
      conv1d_kernel_size: 4
      qkv_proj_blocksize: 4      
  context_length: {seq_length}
  vocab_size: {dataset.tokenizer.get_vocab_size()}
training:
  batch_size: 16
  lr: 0.001
  seed: 42
  val_every_step: 200
  lr_warmup_steps: 2000
  lr_decay_until_steps: 500
  lr_decay_factor: 0.001
  weight_decay: 0.1
  num_steps: 500
  device: mps
  amp_precision: bf16
  weight_precision: float32
  enable_mixed_precision: true
"""

cfg_json = OmegaConf.create(xlstm_cfg)
cfg = from_dict(
    data_class=xLSTMLMModelConfig,
    data=OmegaConf.to_container(cfg_json["model"]),
    config=DaciteConfig(strict=True),
)

training_cfg = from_dict(
    data_class=TrainingConfig,
    data=OmegaConf.to_container(cfg_json["training"]),
    config=DaciteConfig(strict=True),
)

mlf_logger.log_hyperparams(cfg_json)

model = xLSTMLitModel(cfg, training_cfg)

trainer = pl.Trainer(
    max_epochs=max_epochs,
    log_every_n_steps=training_cfg.val_every_step,
    accelerator="mps" if torch.backends.mps.is_available() else "cpu",
    precision=training_cfg.amp_precision if training_cfg.enable_mixed_precision else 32,
    enable_model_summary=True,
    logger=mlf_logger
)

# Train the model
with mlflow.start_run() as run:
    trainer.fit(model, dataloader)
    mlflow.pytorch.log_model(model, artifact_path="model")

    # Fetch the auto logged parameters and metrics.
    print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))


