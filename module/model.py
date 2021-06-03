import torch
import torch.nn as nn
import torch.nn.functional as F

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
    
class VAEencoder(nn.Module):
    def __init__(self, input_size, btl_size, step):
        super(VAEencoder, self).__init__()
        self.linear1 = nn.Linear(input_size, input_size - step * 1)
        # self.linear2 = nn.Linear(input_size - step * 1, input_size - step * 2)
        # self.linear3 = nn.Linear(input_size - step * 2, input_size - step * 3)
        # self.linear4 = nn.Linear(input_size - step * 3, input_size - step * 4)
        # self.linear5 = nn.Linear(input_size - step * 4, input_size - step * 5)

        self.mu_layer = nn.Linear(input_size - step * 1, btl_size)
        self.sigma_layer = nn.Linear(input_size - step * 1, btl_size)

        self.normal = torch.distributions.Normal(0, 1)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        # x = F.relu(self.linear2(x))
        # x = F.relu(self.linear3(x))
        # x = F.relu(self.linear4(x))
        # x = F.relu(self.linear5(x))

        mu = self.mu_layer(x)
        sigma = torch.exp(self.sigma_layer(x))
        z = mu + sigma * self.normal.sample(mu.shape)

        return z

class VAE(nn.Module):
    def __init__(self, input_size, btl_size, step):
        super(VAE, self).__init__()
        self.encoder = VAEencoder(input_size, btl_size, step)
        self.decoder = nn.Sequential(
            nn.Linear(btl_size, input_size - step * 1),
            # nn.Linear(input_size - step * 5, input_size - step * 4),
            # nn.Linear(input_size - step * 4, input_size - step * 3),
            # nn.Linear(input_size - step * 3, input_size - step * 2),
            # nn.Linear(input_size - step * 2, input_size - step * 1),
            nn.Linear(input_size - step * 1, input_size)
        )

    def forward(self, x):
        x = self.encoder(x)
        z = self.decoder(x)
        return z