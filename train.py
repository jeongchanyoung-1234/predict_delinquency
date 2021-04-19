import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from module.model import AutoEncoder
from module.trainer import Trainer
from module.data_loader import get_loaders, get_data

def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)
    p.add_argument('--objective', type=str, required=True)

    p.add_argument('--gpu_id', type=int, default= -1 if torch.cuda.is_available() else 0)
    p.add_argument('--train_ratio', type=float, default=.8)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--n_epochs', type=int, default=20)
    p.add_argument('--verbose', type=int, default=2)

    p.add_argument('--imp', action='store_true')

    ## ae
    p.add_argument('--input_size', type=int, default=44)
    p.add_argument('--step', type=int, default=2)
    p.add_argument('--btl_size', type=int, default=6)

    config = p.parse_args()

    return config

def main(config):

    x, y, _ = get_data(config)
    train_loader, val_loader, test_loader = get_loaders(config, x, y)


    if config.objective == 'ae':

        model = AutoEncoder(config.input_size, config.btl_size, config.step)
        loss = nn.MSELoss()
        optimizer = optim.Adam(model.parameters())

        trainer = Trainer(config)
        trainer.train(model, optimizer, loss, train_loader, val_loader)

    torch.save({
        'model': model.state_dict(),
        'config': config,
    }, config.model_fn)


if __name__ == '__main__':
    config = define_argparser()
    main(config)

