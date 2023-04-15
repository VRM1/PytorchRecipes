import torch
from torch.utils.data import DataLoader, Subset
from pytorch_lightning import LightningModule, Trainer
import pytorch_lightning as pl
import torch.nn.functional as F

# create a toy dataset with 100 samples
dataset = torch.randn(100, 3)

# create three different subsets of the dataset
subset_a = Subset(dataset, range(0, 30))
subset_b = Subset(dataset, range(30, 60))
subset_c = Subset(dataset, range(60, 100))

# create three different data loaders for each subset
train_loader_a = DataLoader(subset_a, batch_size=16, shuffle=True)
train_loader_b = DataLoader(subset_b, batch_size=16, shuffle=True)
train_loader_c = DataLoader(subset_c, batch_size=16, shuffle=True)

# create a validation set
val_dataset = Subset(dataset, range(80, 100))
val_loader = DataLoader(val_dataset, batch_size=16)

# create a test set
test_dataset = Subset(dataset, range(60, 80))
test_loader = DataLoader(test_dataset, batch_size=16)

train_loaders = [train_loader_a, train_loader_b, train_loader_c]

class ToyModel(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        loss = F.mse_loss(output, target)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
        return optimizer



model = ToyModel(input_dim=4, hidden_dim=8, output_dim=2)
trainer = pl.Trainer(max_epochs=10)

for epoch in range(trainer.max_epochs):
    # train loop
    model.train()
    for train_loader in train_loaders:
        for batch_idx, batch in enumerate(train_loader):
            trainer.fit(model, batch)
    
    # validation loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            output = model(data)
            val_loss += F.mse_loss(output, target, reduction='sum').item()
    val_loss /= len(val_dataset)
    
    # test loop
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            output = model(data)
            test_loss += F.mse_loss(output, target, reduction='sum').item()
    test_loss /= len(test_dataset)
    
    print(f"Epoch {epoch+1}: val_loss={val_loss:.4f}, test_loss={test_loss:.4f}")