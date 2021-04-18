import torch.nn as nn


class AutoEncoder(nn.Module):

    def __init__(self, input_size, btl_size, step):
        ### output size shoule be less than 6 ###

        self.btl_size = btl_size

        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, input_size - step),
            nn.ReLU(),

            nn.BatchNorm1d(input_size - step),
            nn.Linear(input_size - step, input_size - step * 2),
            nn.ReLU(),

            nn.BatchNorm1d(input_size - step * 2),
            nn.Linear(input_size - step * 2, btl_size)
        )

        self.decoder = nn.Sequential(
            nn.Linear(btl_size, input_size - step * 2),
            nn.ReLU(),

            nn.BatchNorm1d(input_size - step * 2),
            nn.Linear(input_size - step * 2, input_size - step),
            nn.ReLU(),

            nn.BatchNorm1d(input_size - step),
            nn.Linear(input_size - step, input_size),
        )

    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)

        return y