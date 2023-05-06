import torch
import os
import datetime

from os.path import join, isfile, getmtime, exists

from model import UNet


def save_model(model, optimizer, config):
    if not exists(config.model_path):
        os.makedirs(config.model_path)

    # get the time now
    time_now = datetime.datetime.now()
    time_now = time_now.strftime("%d%b_%H%M")

    # save the model
    filepath = join(config.model_path, f"{config.model_name}_{time_now}.p")

    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'config': config,
    }, filepath)


def create_model(config, load=False, lr=False):
    if not exists(config.model_path):
        os.makedirs(config.model_path)

    files = [join(config.model_path, f) for f in os.listdir(
        config.model_path) if isfile(join(config.model_path, f))]
    files = sorted(files, key=getmtime)
    model = UNet(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    if not load or len(files) == 0:
        config.current_epoch = 0
        return model, config, optimizer

    filepath = files[-1]
    print(f'Load model: {filepath}')
    loaded = torch.load(filepath)

    # copy the config file
    change_config = ("data_length", "data_targetSD", "model_layers",
                     "model_out", "model_kernel", "model_scale")
    for argument in change_config:
        value = getattr(loaded['config'], argument)
        setattr(config, argument, value)

    model = UNet(config)
    model.load_state_dict(loaded['model'])
    if lr:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer.load_state_dict(loaded['optimizer'])

    return model, config, optimizer
