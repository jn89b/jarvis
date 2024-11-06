"""
A script to figure out how to use einops
https://einops.rocks/1-einops-basics/

https://medium.com/@kyeg/einops-in-30-seconds-377a5f4d641a
"""

# rearrange
import einops
import torch
from einops import rearrange

# create a random 4d
batch: int = 3
channels: int = 2
height: int = 4
width: int = 5

tensor = torch.randn(batch, channels, height, width)

# we want to move the channels to the end
# we can do this with rearrange
new_shape = rearrange(tensor, 'b c h w -> b h w c')
assert new_shape.shape == (batch, height, width, channels)
print(new_shape.shape)

# lets try another example
new_tensor = rearrange(tensor, 'b c h w -> b h (c w)')
print(new_tensor.shape)


# Sample data to test (random data for demonstration, adjust as needed for real cases)
idx = [torch.rand(3, 5, 10), torch.rand(2, 5, 10)]  # Example list of tensors
target = [torch.rand(3, 5, 10), torch.rand(2, 5, 10)]  # Example target tensors

# Dummy function to simulate padding (replace with the real function in your code)


def pad_sequence_batch(batch):
    max_len = max(x.size(0) for x in batch)
    return torch.stack([torch.cat([x, torch.zeros(max_len - x.size(0), *x.shape[1:])], dim=0) for x in batch])


# Original method
x_batched = torch.cat(idx, dim=0)
input_batch = pad_sequence_batch([x_batched])
input_batch_type = input_batch[:, :, 0]
input_batch_data = input_batch[:, :, 1:]

# Target processing
y_batched = torch.cat(target, dim=0)
output_batch = pad_sequence_batch([y_batched])
output_batch_type = output_batch[:, :, 0]
output_batch_data = output_batch[:, :, 1:]

# Masks
car_mask_original = (input_batch_type == 1).unsqueeze(-1)
route_mask_original = (input_batch_type == 2).unsqueeze(-1)
other_mask_original = torch.logical_and(
    route_mask_original.logical_not(), car_mask_original.logical_not())
masks_original = [car_mask_original, route_mask_original, other_mask_original]


# New method with einops
# Concatenate and pad sequences for input and output
x_batched_einops = torch.cat(idx, dim=0)
input_batch_einops = pad_sequence_batch([x_batched_einops])
input_batch_type_einops = einops.rearrange(
    input_batch_einops[:, :, 0], 'b s -> b s 1')
input_batch_data_einops = input_batch_einops[:, :, 1:]

y_batched_einops = torch.cat(target, dim=0)
output_batch_einops = pad_sequence_batch([y_batched_einops])
output_batch_type_einops = einops.rearrange(
    output_batch_einops[:, :, 0], 'b s -> b s 1')
output_batch_data_einops = output_batch_einops[:, :, 1:]

# Create masks with einops
car_mask_einops = einops.rearrange(
    input_batch_type_einops == 1, 'b s 1 -> b s 1')
route_mask_einops = einops.rearrange(
    input_batch_type_einops == 2, 'b s 1 -> b s 1')
other_mask_einops = torch.logical_and(~route_mask_einops, ~car_mask_einops)
masks_einops = [car_mask_einops, route_mask_einops, other_mask_einops]

# Verification
print("Verifying batch data equality...")
print(torch.allclose(input_batch, input_batch_einops), "for input batch")
print(torch.allclose(output_batch, output_batch_einops), "for output batch")

print("\nVerifying input batch type equality...")
print(torch.equal(input_batch_type, input_batch_type_einops.squeeze(-1)),
      "for input batch type")

print("\nVerifying input batch data equality...")
print(torch.allclose(input_batch_data, input_batch_data_einops),
      "for input batch data")

print("\nVerifying mask equality...")
print(torch.equal(car_mask_original, car_mask_einops), "for car mask")
print(torch.equal(route_mask_original, route_mask_einops), "for route mask")
print(torch.equal(other_mask_original, other_mask_einops), "for other mask")
