# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# import torch.nn as nn
#
# class ResidualBlock(nn.Module):
#     def __init__(self, in_features, out_features):
#         super(ResidualBlock, self).__init__()
#         self.block = nn.Sequential(
#             nn.Linear(in_features, out_features),
#             nn.BatchNorm1d(out_features),
#             nn.ReLU(inplace=True),
#             nn.Linear(out_features, in_features),
#             nn.BatchNorm1d(in_features)
#         )
#
#     def forward(self, x):
#         return x + self.block(x)  # Add the input x to the output of the block
#
# class Generator(nn.Module):
#     def __init__(self):
#         super(Generator, self).__init__()
#         self.initial = nn.Sequential(
#             nn.Linear(7, 64),
#             nn.BatchNorm1d(64),
#             nn.ReLU(inplace=True)
#         )
#
#         # Example with varied residual block sizes
#         self.residual_blocks = nn.Sequential(
#             ResidualBlock(64, 128),  # First block expands then compresses features
#             ResidualBlock(64, 64),   # Second block maintains the feature size
#             # Add more blocks as needed, adjusting in_features and out_features accordingly
#             # For example, if you want to further compress or expand the feature representation
#             ResidualBlock(64, 32),   # Compresses features
#             ResidualBlock(64, 64)    # Another block that maintains the feature size
#             # Note: Ensure that the output of one block matches the input size of the next
#         )
#
#         self.final = nn.Sequential(
#             nn.Linear(64, 7),
#             nn.Tanh()
#         )
#
#     def forward(self, x):
#         x = self.initial(x)
#         x = self.residual_blocks(x)
#         x = self.final(x)
#         return x


import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(inplace=True),
            nn.Linear(out_features, in_features),
            nn.BatchNorm1d(in_features)
        )

    def forward(self, x):
        return x + self.block(x)  # Residual connection


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.initial = nn.Sequential(
            nn.Linear(10, 64),  # Sentinel-2 has 10 input bands
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )

        self.residual_blocks = nn.Sequential(
            ResidualBlock(64, 128),  # Feature expansion
            ResidualBlock(64, 64),  # Maintain size
            ResidualBlock(64, 32),  # Feature compression
            ResidualBlock(64, 64)  # Final refinement
        )

        self.final = nn.Sequential(
            nn.Linear(64, 10),  # Output 10 bands for bare soil
            nn.Tanh()
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.residual_blocks(x)
        x = self.final(x)
        return x

