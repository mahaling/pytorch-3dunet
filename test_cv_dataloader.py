import importlib
import sys
import torch

from datasets.cloud_volume import get_train_loaders
from unet3d.config import load_config
from unet3d.model import get_model
from unet3d.utils import get_logger, get_tensorboard_formatter
from unet3d.utils import get_number_of_learnable_parameters

def main():
    logger = get_logger('UNet3DTrainer')

    config = load_config()
    logger.info(config)

    manual_seed = config.get('manual_seed', None)
    if manual_seed is not None:
        logger.info(f'Seed the RNG for all devices with {manual_seed}')
        torch.manual_seed(manual_seed)
        # see https://pytorch.org/docs/stable/notes/randomness.html
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        loaders = get_train_loaders(config)

        print(loaders['train'].__len__())

        for i, t in enumerate(loaders['train']):
            for tt in t:
                print(i, tt.shape)


if __name__ == "__main__":
    main()