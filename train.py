import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from module.model import AutoEncoder
from module.trainer import Trainer
from module.data_loader import get_loaders, get_data
from module.classify import classify

def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--objective', type=str, required=True)
    p.add_argument('--save_fn', type=str, required=True)

    # data
    p.add_argument('--imp', action='store_true')

    ## ae
    p.add_argument('--input_size', type=int, default=44)
    p.add_argument('--step', type=int, default=2)
    p.add_argument('--btl_size', type=int, default=6)
    p.add_argument('--gpu_id', type=int, default=-1 if torch.cuda.is_available() else 0)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--n_epochs', type=int, default=20)
    p.add_argument('--verbose', type=int, default=2)

    ## clf
    p.add_argument('--model_name', type=str)
    p.add_argument('--k', type=int)


    config = p.parse_args()

    return config

def main(config):

    if config.objective == 'ae' :
        x, _ = get_data(config)
        train_loader = get_loaders(config, x)

        model = AutoEncoder(config.input_size, config.btl_size, config.step)
        loss = nn.MSELoss()
        optimizer = optim.Adam(model.parameters())

        trainer = Trainer(config)
        trainer.train(model, optimizer, loss, train_loader)

        torch.save({
            'model': model.state_dict(),
            'config': config,
        }, config.save_fn)

    if config.objective == 'clf':
        train, train_df_target, test, new_df, enc_list = get_data(config)
        classify(train, test, config.k, config.save_fn, config)




if __name__ == '__main__':
    config = define_argparser()
    main(config)

