import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from ray.rllib.utils.annotations import override
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.core.columns import Columns
from ray.rllib.utils.typing import Dict, TensorType, List, ModelConfigDict
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
"""
Attention based policy model to tokenize observation space and use transformer
encoder to teach the agent to attend to the most relevant tokens/attributes
https://medium.com/@kaige.yang0110/ray-rllib-ppo-adction-mask-customized-models-90e7ca9767b1
"""


def tokenize_observation(obs:np.ndarray, 
                         self_state_dim:int, 
                         other_state_dim:int) -> np.ndarray:
    """
    Splits the flat observation vector into tokens.
    
    Args:
        obs (np.ndarray): The flat observation vector.
        self_state_dim (int): Dimension of the agent's own state.
        other_state_dim (int): Dimension for each other-agent token.
        
    Returns:
        np.ndarray: Array of tokens with shape [num_tokens, token_dim].
                    The first token is the self state, and each subsequent token is an other-agent's state.
    """
    # Extract the self state token.
    self_token = obs[:self_state_dim]
    
    # The rest of the observation corresponds to other agents.
    remaining = obs[self_state_dim:]
    # Calculate the number of other-agent tokens.
    num_other = len(remaining) // other_state_dim
    
    # Create a list of tokens.
    other_tokens = [
        remaining[i * other_state_dim : (i + 1) * other_state_dim]
        for i in range(num_other)
    ]
    
    # Combine into a single tokens array.
    tokens = [self_token] + other_tokens
    return np.array(tokens)

import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.relu

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # src shape: [L, B, E]
        attn_output, attn_weights = self.self_attn(
            src, src, src,
            need_weights=True,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn_weights

class MultiTransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, nhead, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        attn_weights_all = []
        output = src
        for layer in self.layers:
            output, attn_weights = layer(output, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
            attn_weights_all.append(attn_weights)
        return output, attn_weights_all

class TransformerEncoder(nn.Module):
    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        self_state_dim: int = 10,   # number of features for self state
        token_dim: int = 5,         # number of features per relative token
        embedding_dim: int = 64,  # embedding dimension for transformer
        num_layers: int = 2,
        nhead: int = 2,
    ):
        """
        obs_space: The observation space containing "observations" and "action_mask".
        action_space: The action space.
        num_outputs: The number of outputs (logits) for the policy head.
        
        The idea of this encoder is to tokenize the observation space into
        the following:
            - self token: the agent's own state
            - relative tokens: the states of other agents
            
        The attention map should look like this:
        - token 0 (self token) should attend to all other tokens
        - token 1 (other agent 1) Which are the features of the other agent 1
        - token 2 (other agent 2) Which are the features of the other agent 2
        - token n (other agent n) Which are the features of the other agent n
        
        The transformer encoder will then learn to attend to the most relevant
        tokens/attributes.
    
        """
        super().__init__()
        self.self_state_dim = self_state_dim
        self.token_dim = token_dim

        # Determine the full observation dimension.
        # We assume that obs_space["observations"].shape[0] is the flat obs dim.
        full_obs_dim = int(np.prod(obs_space["observations"].shape))
        # The first self_state_dim elements correspond to the self token.
        # The remaining elements represent relative tokens.
        remaining = full_obs_dim - self_state_dim
        # Compute the number of relative tokens.
        self.num_relative_tokens = remaining // token_dim
        
        # Total number of tokens: self token + relative tokens.
        self.num_tokens = 1 + self.num_relative_tokens

        ## Remember we go from low dimension to high dimension then back to low dimension
        self.embedding_dim = embedding_dim
        # Embedding for the self token:
        self.self_embedding = nn.Linear(self_state_dim, embedding_dim)
        # Embedding for relative tokens:
        self.token_embedding = nn.Linear(token_dim, embedding_dim)
        
        # Transformer encoder:          
        self.transformer, self.attn_weights = None, None
        self.custom_transformer = MultiTransformerEncoder(
            num_layers, embedding_dim, nhead, dropout=0.1)
        self.policy_head = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_outputs)
        )
        
    def forward(self, input_dict):
        """
        Expects input_dict["obs"] to be a tensor of shape [B, full_obs_dim].
        The observation is a dict with keys "observations" and "action_mask" before _preprocess_batch,
        but here only the flat "observations" are passed in.
        """
        #obs = input_dict["obs"].float()  # shape: [B, full_obs_dim]
        obs = input_dict.float()
        B = obs.size(0)

        # Tokenize the observation:
        # 1. Self token: first self_state_dim features.
        self_token = obs[:, :self.self_state_dim]  # shape: [B, self_state_dim]
        # 2. Relative tokens: remaining features.
        relative_obs = obs[:, self.self_state_dim:]  # shape: [B, num_relative_tokens * token_dim]
        # Reshape into tokens: [B, num_relative_tokens, token_dim]
        relative_tokens = relative_obs.view(B, self.num_relative_tokens, self.token_dim)

        # Embed tokens:
        self_token_embed = self.self_embedding(self_token).unsqueeze(1)  # [B, 1, embedding_dim]
        relative_token_embed = self.token_embedding(relative_tokens)       # [B, num_relative_tokens, embedding_dim]

        # Concatenate to form the full sequence: [B, num_tokens, embedding_dim]
        tokens = torch.cat([self_token_embed, relative_token_embed], dim=1)

        # Transformer expects shape [sequence_length, batch_size, embedding_dim]
        # tokens = tokens.transpose(0, 1)  # now shape: [num_tokens, B, embedding_dim]
        # transformer_out = self.transformer(tokens)  # shape: [num_tokens, B, embedding_dim]
        # transformer_out = transformer_out.transpose(0, 1)  # back to [B, num_tokens, embedding_dim]
        tokens = tokens.transpose(0, 1)
        transformer_out, attn_weights_all = self.custom_transformer(tokens)
        transformer_out = transformer_out.transpose(0, 1)  # back to [B, num_tokens, embedding_dim]

        # Pool over tokens (mean pooling here) to get a fixed-size representation.
        pooled = transformer_out.mean(dim=1)  # shape: [B, embedding_dim]

        # Generate action logits.
        logits = self.policy_head(pooled)  # shape: [B, num_outputs]

        return {"logits": logits, "attn_weights": attn_weights_all}