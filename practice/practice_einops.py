"""
A script to figure out how to use einops
https://einops.rocks/1-einops-basics/

https://medium.com/@kyeg/einops-in-30-seconds-377a5f4d641a
"""

# rearrange
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
