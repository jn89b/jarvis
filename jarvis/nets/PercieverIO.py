from typing import Callable, Tuple
from gymnasium import spaces
import torch as th
from torch import nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy


class PerceiverIO(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, num_heads: int, num_latents: int, output_dim: int):
        super(PerceiverIO, self).__init__()
        # Project input_dim to match latent_dim
        self.input_projection = nn.Linear(input_dim, latent_dim)
        
        self.latent_array = nn.Parameter(th.randn(num_latents, latent_dim))
        self.cross_attention = nn.MultiheadAttention(latent_dim, num_heads)
        self.latent_processing = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )
        self.decoder = nn.Linear(latent_dim, output_dim)

    def forward(self, x: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        # Project input to match latent dimension
        x_projected = self.input_projection(x).unsqueeze(1)  # Add a sequence dimension
        # Cross-attention: map input to latent space and get attention weights
        latents, attention_weights = self.cross_attention(self.latent_array.unsqueeze(1), x_projected, x_projected)
        latents = self.latent_processing(latents)
        output = self.decoder(latents)
        return output.mean(dim=0), attention_weights

class CustomNetwork(nn.Module):
    def __init__(self, feature_dim: int, last_layer_dim_pi: int = 64, last_layer_dim_vf: int = 64):
        super(CustomNetwork, self).__init__()

        # Define latent dimension (for Stable Baselines 3 compatibility)
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        
        # Policy network (Perceiver IO)
        self.policy_net = PerceiverIO(feature_dim, latent_dim=256, num_heads=8, num_latents=10, output_dim=last_layer_dim_pi)

        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_vf), nn.ReLU()
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        action, _ = self.policy_net(features)
        return action

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)

class PercieverIOPolicy(ActorCriticPolicy):
    def __init__(self, observation_space: spaces.Space, 
                 action_space: spaces.Space, 
                 lr_schedule: Callable[[float], float], *args, **kwargs):
        kwargs["ortho_init"] = False
        super(PercieverIOPolicy, self).__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.features_dim)

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor]:
        return super().forward(obs, deterministic)

# Train the model with Perceiver IO-based policy
#use gpu

# model = PPO(PercieverIOPolicy, "CartPole-v1", verbose=1)
# model.learn(total_timesteps=300000)
# # Save the model
# model.save("ppo_perceiverio_cartpole")


# model = PPO("MlpPolicy", "CartPole-v1", verbose=1)
# model.learn(total_timesteps=100000)
# # Save the model
# model.save("ppo_mlp_cartpole")

