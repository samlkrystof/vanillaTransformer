import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader
import pandas as pd

from VanillaTransformerModel import Transformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load data
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

# environment variable with wandb api key must be set
wandb.login()
run = wandb.init(project="VanillaTransformer")

# create dataset
train_dataset = VanillaTransformerDataset(train_df)

# create dataloader
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# create model
model = Transformer(4, 6, 512, 8, 1024, 0.1,64, "gelu").to(device)

# create optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# create loss function
criterion = nn.MSELoss()

wandb.watch(model, criterion, log="all", log_freq=100)

# training loop
min_loss = 1e12
for epoch in range(10):
    for batch in tqdm.tqdm(train_dataloader):
        optimizer.zero_grad()

        input = batch['input'].to(device)
        target = batch['target'].to(device)

        output = model(input)

        loss = criterion(output, target)
        loss.backward()

        optimizer.step()

        wandb.log({"loss": loss.item()})

    if loss.item() < min_loss:
        min_loss = loss.item()
        torch.save(model.state_dict(), "model.pt")

    print(f"Epoch: {epoch}, Loss: {loss.item()}")
    wandb.log({"epoch": epoch, "loss": loss.item()})







