import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomNetwork(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        # Features_dim is the number of output features
        super(CustomNetwork, self).__init__(observation_space, features_dim)
        self.net = nn.Sequential(
            nn.Linear(observation_space.shape[0], 128),  # First layer
            nn.ReLU(),
            nn.Linear(128, 256),  # Second layer
            nn.ReLU(),
            nn.Linear(256, features_dim),  # Output layer
            nn.ReLU()
        )
    
    def forward(self, observations):
        return self.net(observations)
    

from stable_baselines3.common.policies import ActorCriticPolicy

class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomActorCriticPolicy, self).__init__(
            *args,
            **kwargs,
            features_extractor_class=CustomNetwork,
            features_extractor_kwargs=dict(features_dim=256),
        )