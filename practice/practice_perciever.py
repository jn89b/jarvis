from transformers import PerceiverModel, PerceiverConfig
import torch
import torch.nn as nn


class EvaderFormer(nn.Module):
    def __init__(self, perceive_config: PerceiverConfig) -> None:
        super().__init__()
        self.model: PerceiverModel = PerceiverModel(perceive_config)

    def forward(self, pursuer_features, ego_features):
        """
        Forward method to calculate attention values between pursuers and ego.

        Args:
            pursuer_features (torch.Tensor): Features of pursuers of shape (batch_size, num_pursuers, num_features).
            ego_features (torch.Tensor): Features of the ego of shape (batch_size, 1, num_features).

        Returns:
            attention_values (torch.Tensor): Attention scores between ego and pursuers.
        """
        # Combine ego and pursuers for model input
        # Shape: (batch_size, num_pursuers + 1, num_features)
        inputs = torch.cat([ego_features, pursuer_features], dim=1)

        # Pass through Perceiver model
        perceiver_output = self.model(inputs=inputs)

        # Extract attention weights from the model
        # Assuming the model provides attention weights
        attention_values = perceiver_output.attentions

        # Optionally, extract the attention for the ego (first element) with respect to pursuers
        # Attention of ego to each pursuer
        ego_attention_values = attention_values[:, 0, 1:]

        return ego_attention_values


# Example setup
num_pursuers = 5
num_features = 6  # x, y, psi, v, vx, vy

# Define model configuration
# Define model configuration with d_model matching num_features
config = PerceiverConfig(
    d_model=num_features,          # Match input features dimension
    input_channels=num_features,    # Dimensionality of each pursuer's feature vector
    num_latents=16,                 # Number of latent vectors for information aggregation
    d_latents=256,                  # Dimensionality of each latent vector
    num_cross_attention_heads=4,    # Number of attention heads for cross-attention
    num_self_attention_heads=4,     # Number of attention heads for self-attention
    num_self_attention_layers=2,    # Number of self-attention layers
    output_channels=1,              # Single output per pursuer for relevance scoring
    output_shape=(num_pursuers + 1,)
)


# Initialize model
model = EvaderFormer(config)

# Example inputs
batch_size = 2
pursuer_features = torch.randn(batch_size, num_pursuers, num_features)
ego_features = torch.randn(batch_size, 1, num_features)

# Forward pass to get attention values
attention_values = model(pursuer_features, ego_features)
print("Attention Values (ego to pursuers):", attention_values)
