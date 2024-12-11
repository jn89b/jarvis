import torch
import torch.nn as nn

# Flow Feature Embedding (CNN)


class FlowFeatureCNN(nn.Module):
    def __init__(self, input_channels, embed_dim):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, embed_dim, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d(1)  # Reduce spatial dimensions
        )

    def forward(self, flow_input):
        # Flatten to [B, embed_dim]
        return self.cnn(flow_input).squeeze(-1).squeeze(-1)

# Multi-Head Attention


class AttentionModule(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, query, key, value):
        attn_output, _ = self.attention(query, key, value)
        return attn_output

# Transformer Policy Network


class PolicyNetwork(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, action_dim):
        super().__init__()
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4
        )
        self.transformer = nn.TransformerEncoder(
            self.transformer_layer, num_layers=num_layers)
        self.policy_head = nn.Linear(embed_dim, action_dim)
        self.value_head = nn.Linear(embed_dim, 1)

    def forward(self, encoded_features):
        # Transformer for temporal encoding
        transformer_output = self.transformer(encoded_features)
        # Compute policy and value
        policy = self.policy_head(transformer_output[-1])  # Last time step
        value = self.value_head(transformer_output[-1])
        return policy, value

# Full RL Model


class MarineFormerRL(nn.Module):
    def __init__(self, state_dim, flow_input_dim, embed_dim, num_heads, num_layers, action_dim):
        super().__init__()
        self.static_embed = nn.Linear(state_dim, embed_dim)
        self.dynamic_embed = nn.Linear(state_dim, embed_dim)
        self.flow_cnn = FlowFeatureCNN(flow_input_dim, embed_dim)
        self.attention_static = AttentionModule(embed_dim, num_heads)
        self.attention_dynamic = AttentionModule(embed_dim, num_heads)
        self.policy_network = PolicyNetwork(
            embed_dim, num_heads, num_layers, action_dim)

    def forward(self, static_state, dynamic_state, flow_state):
        # Embed features
        static_features = self.static_embed(static_state)
        dynamic_features = self.dynamic_embed(dynamic_state)
        flow_features = self.flow_cnn(flow_state)

        # Attention over static and dynamic states
        static_attention = self.attention_static(
            static_features, static_features, static_features)
        dynamic_attention = self.attention_dynamic(
            dynamic_features, dynamic_features, dynamic_features)

        # Combine features
        combined_features = torch.cat(
            [static_attention, dynamic_attention, flow_features.unsqueeze(0)], dim=0)

        # Policy and value computation
        policy, value = self.policy_network(combined_features)
        return policy, value


# Example Input Data
batch_size = 8
state_dim = 10
flow_input_dim = 3  # Channels of flow input (e.g., RGB or flow map)
embed_dim = 64
num_heads = 4
num_layers = 2
action_dim = 5

static_state = torch.rand(batch_size, state_dim)
dynamic_state = torch.rand(batch_size, state_dim)
flow_state = torch.rand(batch_size, flow_input_dim, 64, 64)  # Flow map

# Instantiate and run the model
model = MarineFormerRL(state_dim, flow_input_dim,
                       embed_dim, num_heads, num_layers, action_dim)
policy, value = model(static_state, dynamic_state, flow_state)

print("Policy logits:", policy)
print("State value:", value)
