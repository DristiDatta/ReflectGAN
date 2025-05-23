import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.linear = nn.Sequential(
            nn.Linear(14, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),  # Adding Dropout
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128),  # Adding Batch Normalization
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Sigmoid activation for binary classification
        )

    def forward(self, input, target):
        input_target = torch.cat((input, target), 1)
        return self.linear(input_target)
