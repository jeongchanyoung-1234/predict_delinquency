import torch.nn as nn


class AutoEncoder(nn.Module):

    def __init__(self, input_size, btl_size, step) :
        self.input_size = input_size
        self.btl_size = btl_size
        self.step = step

        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, input_size - step),
            nn.ReLU(),

            nn.BatchNorm1d(input_size - step),
            nn.Linear(input_size - step, input_size - step * 2),
            nn.ReLU(),

            nn.BatchNorm1d(input_size - step * 2),
            nn.Linear(input_size - step * 2, input_size - step * 3),
            nn.ReLU(),

            nn.BatchNorm1d(input_size - step * 3),
            nn.Linear(input_size - step * 3, input_size - step * 4),
            nn.ReLU(),

            nn.BatchNorm1d(input_size - step * 4),
            nn.Linear(input_size - step * 4, input_size - step * 5),
            nn.ReLU(),

            nn.BatchNorm1d(input_size - step * 5),
            nn.Linear(input_size - step * 5, btl_size)
        )

        self.decoder = nn.Sequential(
            nn.Linear(btl_size, input_size - step * 5),
            nn.ReLU(),

            nn.BatchNorm1d(input_size - step * 5),
            nn.Linear(input_size - step * 5, input_size - step * 4),
            nn.ReLU(),

            nn.BatchNorm1d(input_size - step * 4),
            nn.Linear(input_size - step * 4, input_size - step * 3),
            nn.ReLU(),

            nn.BatchNorm1d(input_size - step * 3),
            nn.Linear(input_size - step * 3, input_size - step * 2),
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