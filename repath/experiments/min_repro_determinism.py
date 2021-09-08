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
from torchvision.datasets import ImageFolder, MNIST
from torchvision.models import GoogLeNet, googlenet
from torchvision.transforms import Compose, ToTensor, RandomCrop, RandomRotation


global_seed = 123
experiment_root = Path(__file__).parent.parent.parent / "experiments" / "repro_test"

torch.manual_seed(global_seed)
random.seed(global_seed)
np.random.seed(global_seed)
# np random number generators???
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic=True
torch.use_deterministic_algorithms(True)

torch.cuda.manual_seed(global_seed)
torch.cuda.manual_seed_all(global_seed)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    torch.manual_seed(worker_seed)
    torch.cuda.manual_seed(worker_seed)
    torch.cuda.manual_seed_all(global_seed)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.use_deterministic_algorithms(True)


class PatchClassifier1(pl.LightningModule):
    
    def __init__(self) -> None:
        super().__init__()
        self.layer_1 = nn.Linear(224 * 224 * 3, 128)
        self.layer_2 = nn.Linear(128, 256)
        self.layer_3 = nn.Linear(256, 10)

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        pred = torch.log_softmax(output, dim=1)

        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        
        correct=pred.argmax(dim=1).eq(y).sum().item()
        total=len(y)   
        accu = correct / total
        self.log("train_accuracy", accu, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), 
                                    lr=0.01,
                                    weight_decay=0.0005)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.StepLR(optimizer, step_size=50000, gamma=0.5),
            'interval': 'step' 
        }
        return [optimizer], [scheduler]

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        x = x.view(batch_size, -1)
        x = self.layer_1(x)
        x = nn.functional.relu(x)
        x = self.layer_2(x)
        x = nn.functional.relu(x)
        x = self.layer_3(x)
        return x


class PatchClassifier2(pl.LightningModule):
    
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(global_seed)
        self.model=googlenet(pretrained=True, aux_logits=True)
        self.model.dropout = nn.Linear(1024,1024)

    def training_step(self, batch, batch_idx):
        x, y = batch
        output, aux1, aux2 = self.model(x)
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

# create a logger
csv_logger1 = pl_loggers.CSVLogger(experiment_root / 'logs', name='basic_classifier2', version=0)
# create a logger
csv_logger2 = pl_loggers.CSVLogger(experiment_root / 'logs', name='patch_classifier2', version=0)

# train our model
classifier2 = PatchClassifier2()
trainer2 = pl.Trainer(gpus=1, accelerator="ddp", max_epochs=3, 
                    logger=csv_logger2, plugins=DDPPlugin(find_unused_parameters=False), deterministic=True)
# train our model
classifier1 = PatchClassifier1()
trainer1 = pl.Trainer(gpus=1, accelerator="ddp", max_epochs=3, 
                    logger=csv_logger1, plugins=DDPPlugin(find_unused_parameters=False), deterministic=True)


transform = Compose([
        #RandomRotateFromList([0.0, 90.0, 180.0, 270.0]),
        #RnadomCropSpecifyOffset(32),
        #RandomCrop((224, 224)),
        #RandomRotation((0,360)),
        ToTensor() #,
    ])

batch_size = 32

# train_set = MNIST(root = experiment_root / "mnisttraining", train=True,  transform=transform, download=False)
train_set = ImageFolder(experiment_root / "training_patches", transform=transform)

# get first 3 items
#three_images_hash = [hash(next(iter(train_set))), hash(next(iter(train_set))), hash(next(iter(train_set)))]
#print(three_images_hash)

# create dataloaders
g = torch.Generator()
g.manual_seed(0)
train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=1, worker_init_fn=seed_worker, generator=g)

print(hash(next(iter(train_loader))[0]))
exit()

# train our model
trainer2.fit(classifier2, train_dataloaders=train_loader)

# train our model
trainer1.fit(classifier1, train_dataloaders=train_loader)

