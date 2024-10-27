import yaml
import torch
from jarvis.datasets.base_dataset import BaseDataset
from torch.utils.data import DataLoader
from jarvis.transformers.evadeformer import EvadeFormer

# if __name__ == "__main__":
data_config = "config/data_config.yaml"
# Load the YAML file

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open(data_config, 'r') as f:
    data_config = yaml.safe_load(f)
batch_size: int = 5
dataset = BaseDataset(config=data_config, is_validation=False)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=dataset.collate_fn)

print("len of dataloader: ", len(dataloader))
model_config = 'config/data_config.yaml'
with open(model_config, 'r') as f:
    model_config = yaml.safe_load(f)

model = EvadeFormer(
    config=model_config
).to(device)
optimizer = model.optimizer

epochs: int = 10
print_every: int = 15

print("Model device:", next(model.parameters()).device)

for epoch in range(epochs):
    model.train()
    i = 0
    for batch in dataloader:
        # for k, v in batch.items():
        #     if isinstance(v, torch.Tensor):
        #         print(f"Batch key '{k}' is on device: {v.device}")
        # Forward pass
        output, loss = model(batch)

        # Zero the gradients, perform the backward pass, and update weights
        optimizer.zero_grad()    # Reset gradients
        loss.backward()          # Compute gradients
        optimizer.step()         # Update weights

        # # get the loss
        # loss = model.loss.item()
        # print(loss)

        if i % print_every == 0:
            print(
                f"Epoch {epoch + 1}/{epochs}, Batch {i}/{len(dataloader)}, Loss: {loss:.4f}")
