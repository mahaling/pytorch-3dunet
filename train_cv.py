import importlib
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from datasets.cloud_volume import get_train_loaders
from unet3d.config import load_config
from unet3d.losses import get_loss_criterion
from unet3d.metrics import get_evaluation_metric
from unet3d.model import get_model
from unet3d.trainer import UNet3DTrainer
from unet3d.utils import get_logger, get_tensorboard_formatter
from unet3d.utils import get_number_of_learnable_parameters

from train import _create_trainer, _create_optimizer, _create_lr_scheduler

def main():
    # Load and log experiment configuration
    config = load_config()
    
    # Create main logger
    logfile = config.get('logfile', None)
    logger = get_logger('UNet3DTrainer', logfile=logfile)

    logger.info(config)

    manual_seed = config.get('manual_seed', None)
    if manual_seed is not None:
        logger.info(f'Seed the RNG for all devices with {manual_seed}')
        torch.manual_seed(manual_seed)
        # see https://pytorch.org/docs/stable/notes/randomness.html
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Create the model
    model = get_model(config)

    # multiple GPUs
    if (torch.cuda.device_count() > 1):
        logger.info("There are {} GPUs available".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    # put the model on GPUs
    logger.info(f"Sending the model to '{config['device']}'")
    model = model.to(config['device'])
    
    # Log the number of learnable parameters
    logger.info(f'Number of learnable params {get_number_of_learnable_parameters(model)}')

    # Create loss criterion
    loss_criterion = get_loss_criterion(config)
    logger.info(f"Created loss criterion: {config['loss']['name']}")
    
    # Create evaluation metric
    eval_criterion = get_evaluation_metric(config)
    logger.info(f"Created eval criterion: {config['eval_metric']['name']}")

    # Create data loaders
    loaders = get_train_loaders(config)

    # Create the optimizer
    optimizer = _create_optimizer(config, model)

    # Create learning rate adjustment strategy
    lr_scheduler = _create_lr_scheduler(config, optimizer)

    # Create model trainer
    trainer = _create_trainer(config, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler,
                              loss_criterion=loss_criterion, eval_criterion=eval_criterion, loaders=loaders,
                              logger=logger)
    
    # Start training
    trainer.fit()

if __name__ == '__main__':
    main()