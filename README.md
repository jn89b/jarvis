# Big Letter 

```python
def adjust_attention_based_on_distance(attn_logits: Tensor, distances: Tensor, temperature: float = 1e-5) -> Tensor:
    """
    Adjusts attention logits by scaling them based on inverse distance to the ego vehicle.

    Args:
        attn_logits (Tensor): Raw attention logits from the transformer, shape [batch_size, num_heads, num_vehicles].
        distances (Tensor): Distances between the ego and each vehicle, shape [batch_size, num_vehicles].
        temperature (float): Temperature scaling to control the sharpness of attention distribution.

    Returns:
        Tensor: Adjusted attention weights, shape [batch_size, num_heads, num_vehicles].
    """
    # Compute inverse of the distances
    inverse_distances = 1 / (distances**2 + 1e-6)  # Prevent division by zero
    # inverse_distances = torch.exp(-distances)

    # Expand dimensions of inverse distances to match attention logits
    # Shape of inverse_distances: [batch_size, num_vehicles] --> [batch_size, 1, num_vehicles]
    inverse_distances = inverse_distances.unsqueeze(1)

    # Multiply inverse distances with the attention logits
    scaled_attn_logits = attn_logits * inverse_distances

    # Apply softmax with temperature scaling
    attention_weights = F.softmax(scaled_attn_logits/temperature, dim=-1)

    return attention_weights
```


# How to push Github stuff through CLI 
```
git add . # add any current revisions from code base
```