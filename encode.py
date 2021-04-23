import argparse

import pandas as pd

import torch

from module.model import AutoEncoder
from module.data_loader import get_data

def define_argparse():
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', type=str, required=True)
    p.add_argument('--save_fn', type=str, required=True)
    p.add_argument('--gpu_id', type=int, default=0 if torch.cuda.is_available() else -1)
    config = p.parse_args()

    return config


def main(config):
    saved_data = torch.load(config.model_fn,
                            map_location='cpu' if config.gpu_id < 0 else 'cuda:{}'.format(config.gpu_id))


    model_dict = saved_data['model']
    train_config = saved_data['config']

    model = AutoEncoder(train_config.input_size, train_config.btl_size, train_config.step)
    model.load_state_dict(model_dict)

    _, X = get_data(train_config)
    result = model.encoder(torch.from_numpy(X).float())
    result = result.detach().numpy()
    print('Result : shape {}'.format(result.shape))
    pd.DataFrame(result).to_csv('data/{}'.format(config.save_fn), index=False)


if __name__ == '__main__':
    config = define_argparse()
    main(config)