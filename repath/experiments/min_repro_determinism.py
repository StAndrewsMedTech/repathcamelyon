from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import seed_everything
from pytorch_lightning.plugins import DDPPlugin
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import GoogLeNet
from torchvision.transforms import Compose, ToTensor, RandomCrop, RandomRotation


global_seed = 123
experiment_root = Path(__file__).parent.parent.parent / "experiments" / "repro_test"

torch.backends.cudnn.deterministic=True
torch.manual_seed(global_seed)
torch.cuda.manual_seed(global_seed)
np.random.seed(global_seed)
random.seed(global_seed)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    torch.backends.cudnn.deterministic=True
    torch.manual_seed(worker_seed)
    torch.cuda.manual_seed(worker_seed)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class PatchClassifier(pl.LightningModule):
    
    def __init__(self) -> None:
        super().__init__()
        self.model = GoogLeNet(num_classes=2, init_weights=True)
        self.model.dropout = nn.Dropout(0.5)

    def training_step(self, batch, batch_idx):
        x, y = batch
        output, aux2, aux1 = self.model(x)
        pred = torch.log_softmax(output, dim=1)

        criterion = nn.CrossEntropyLoss()
        loss1 = criterion(output, y)
        loss2 = criterion(aux1, y)
        loss3 = criterion(aux2, y)
        loss = loss1 + 0.3 * loss2 + 0.3 * loss3
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        
        correct=pred.argmax(dim=1).eq(y).sum().item()
        total=len(y)   
        accu = correct / total
        self.log("train_accuracy", accu, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self.model(x)
        pred = torch.log_softmax(output, dim=1)

        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, y)
        self.log("val_loss", loss, on_epoch=True, sync_dist=True)
        
        correct=pred.argmax(dim=1).eq(y).sum().item()
        total=len(y)   
        accu = correct / total
        self.log("val_accuracy", accu, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), 
                                    lr=0.01,
                                    weight_decay=0.0005)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.StepLR(optimizer, step_size=50000, gamma=0.5),
            'interval': 'step' 
        }
        return [optimizer], [scheduler]

    def forward(self, x):
        return self.model(x)



transform = Compose([
        #RandomRotateFromList([0.0, 90.0, 180.0, 270.0]),
        #RnadomCropSpecifyOffset(32),
        RandomCrop((224, 224)),
        RandomRotation((0,360)),
        ToTensor() #,
    ])

batch_size = 32

train_set = ImageFolder(experiment_root / "training_patches", transform=transform)
valid_set = ImageFolder(experiment_root / "validation_patches", transform=transform)

# create dataloaders
g = torch.Generator()
g.manual_seed(0)
train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=8, worker_init_fn=seed_worker, generator=g)
valid_loader = DataLoader(valid_set, batch_size=batch_size, num_workers=8, worker_init_fn=seed_worker, generator=g)

# configure logging and checkpoints
checkpoint_callback = ModelCheckpoint(
    monitor="val_accuracy",
    dirpath=experiment_root / "patch_model",
    filename=f"checkpoint",
    mode="max",
)

# create a logger
csv_logger = pl_loggers.CSVLogger(experiment_root / 'logs', name='patch_classifier', version=0)

# train our model
#torch.manual_seed(global_seed)
classifier = PatchClassifier()
trainer = pl.Trainer(callbacks=[checkpoint_callback], gpus=8, accelerator="ddp", max_epochs=3, 
                    logger=csv_logger, plugins=DDPPlugin(find_unused_parameters=False), deterministic=True)
trainer.fit(classifier, train_dataloaders=train_loader, val_dataloaders=valid_loader)


# configure logging and checkpoints
checkpoint_callback = ModelCheckpoint(
    monitor="val_accuracy",
    dirpath=experiment_root / "patch_model2",
    filename=f"checkpoint",
    mode="max",
)

# create a logger
csv_logger = pl_loggers.CSVLogger(experiment_root / 'logs', name='patch_classifier2', version=0)

# train our model
#torch.manual_seed(global_seed)
classifier = PatchClassifier()
trainer = pl.Trainer(callbacks=[checkpoint_callback], gpus=8, accelerator="ddp", max_epochs=3, 
                    logger=csv_logger, plugins=DDPPlugin(find_unused_parameters=False), deterministic=True)
trainer.fit(classifier, train_dataloaders=train_loader, val_dataloaders=valid_loader)