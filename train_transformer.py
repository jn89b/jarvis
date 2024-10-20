import json
import torch
from torch.utils.data import DataLoader, Dataset
from typing import List, Tuple
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter  # Import the SummaryWriter
# Assuming this is your transformer
from jarvis.transformers.test import CarTransformer
from torch.nn.utils import clip_grad_norm_

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to load JSON files


def open_json_file(file_path: str) -> dict:
    with open(file_path) as f:
        data = json.load(f)
    return data

# Dataset class for Car Trajectories


class CarTrajectoryDataset(Dataset):
    def __init__(self, data_files: List[dict]) -> None:
        self.data = data_files  # Each file contains a sequence of time steps

    def __len__(self) -> int:
        return len(self.data)  # Number of sequences

    def __getitem__(self, idx: int) -> List[dict]:
        # Return the entire sequence (multiple time steps)
        return self.data[idx]

# Collate function for padding sequences


def collate_fn(batch: List[List[dict]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ego_seq = [torch.tensor([step['ego'] for step in b],
                            dtype=torch.float32) for b in batch]
    vehicles_seq = [torch.tensor(
        [step['vehicles'] for step in b], dtype=torch.float32) for b in batch]
    waypoints = [torch.tensor([step['future_waypoints']
                              for step in b], dtype=torch.float32) for b in batch]

    # Pad sequences
    padded_ego = pad_sequence(ego_seq, batch_first=True)
    padded_vehicles = pad_sequence(vehicles_seq, batch_first=True)
    padded_waypoints = pad_sequence(waypoints, batch_first=True)

    return padded_ego, padded_vehicles, padded_waypoints


# Load multiple JSON files for the dataset
file_lists = []
for data in range(1):
    file_name = 'data/' + 'simulation_data_' + str(data) + '.json'
    json_data = open_json_file(file_name)
    file_lists.append(json_data)

# Create the dataset and dataloader
dataset = CarTrajectoryDataset(file_lists)  # Example files
batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, collate_fn=collate_fn)

# Initialize your model and move it to GPU
model = CarTransformer().to(device)

# Define loss function and optimizer
loss_fn = torch.nn.L1Loss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Initialize TensorBoard SummaryWriter
writer = SummaryWriter(log_dir='runs/car_transformer_experiment')

# Load checkpoint to resume training if available
checkpoint_path = 'evader_transformer_model.pth'  # Example checkpoint
continue_training = False
start_epoch = 0

if continue_training:
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Resuming training from epoch {start_epoch}")

# Training loop
epochs = 150
print_every = 100
save_every = 10
grad_clip: float = 1.0

for epoch in range(start_epoch, epochs):
    model.train()
    total_loss: float = 0.0

    for batch in dataloader:
        padded_ego, padded_vehicles, waypoints = batch

        # Move the data to the GPU
        padded_ego = padded_ego.to(device)
        padded_vehicles = padded_vehicles.to(device)
        waypoints = waypoints.to(device)

        batch_size, seq_length, _ = padded_ego.shape

        # Loop over each time step in the sequence
        data_len = 100
        for t in range(100):
            ego_data_t = padded_ego[:, t, :]
            vehicles_data_t = padded_vehicles[:, t, :, :]
            waypoints_t = waypoints[:, t, :]
            last_wp = waypoints_t[0, -1, :]  # Example target point

            # Pass only the time step data to the model
            pred_waypoints, attn_weights = model(
                ego_data_t, vehicles_data_t, last_wp)

            # Calculate loss between predicted and actual waypoints
            loss = loss_fn(pred_waypoints, waypoints_t)

            # Backpropagation
            optimizer.zero_grad()  # Clear previous gradients
            loss.backward()  # Backpropagate the error
            # Apply gradient clipping here
            clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()  # Update the model weights

            total_loss += loss.item()

            # Log to TensorBoard
            step = epoch * len(dataloader) + t
            writer.add_scalar('Loss/train', loss.item(), step)

            if t % print_every == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs}, Batch {t}/{seq_length}, Loss: {loss.item():.4f}")

    if epoch % save_every == 0:
        # Save the model every few epochs
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, f'evader_transformer_model_{epoch}.pth')

    # Calculate average loss for this epoch
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")

    # Log average loss for the epoch to TensorBoard
    writer.add_scalar('Loss/average_epoch_loss', avg_loss, epoch)

# Close the writer when done
writer.close()

# save the final model
torch.save({
    'epoch': epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, 'evader_transformer_model.pth')
