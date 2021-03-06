import importlib
import os
import torch
import torch.nn as nn

from datasets.cloud_volume import get_test_loaders
from unet3d import utils
from unet3d.config import load_config
from unet3d.model import get_model

#logger = utils.get_logger('UNet3DPredictor')

threshold = 0.8

def _get_output_file(out_path, filename, suffix='predictions', format='h5'):
    return f"{os.path.join(out_path, '{}_{}.{}'.format(filename, suffix, format))}"


def _get_dataset_names(config, number_of_datasets, prefix='predictions'):
    dataset_names = config.get('test_dataset_name')
    if dataset_names is not None:
        if isinstance(dataset_names, str):
            return [dataset_names]
        else:
            return dataset_names
    else:
        if number_of_datasets == 1:
            return [prefix]
        else:
            return [f'{prefix}{i}' for i in range(number_of_datasets)]


def _get_predictor(model, loader, output_file, config):
    predictor_config = config.get('predictor', {})
    class_name = predictor_config.get('name', 'StandardPredictor')

    m = importlib.import_module('unet3d.predictor')
    predictor_class = getattr(m, class_name)

    return predictor_class(model, loader, output_file, config, **predictor_config)


def main():
    # Load configuration
    config = load_config()

    # create logger
    logfile = config.get('logfile', None)
    logger = utils.get_logger('UNet3DPredictor', logfile=logfile)

    # Create the model
    logger.info(f'Creating the model')
    model = get_model(config)

    # multiple GPUs
    if (torch.cuda.device_count() > 1):
        logger.info("There are {} GPUs available".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    # Load model state
    model_path = config['model_path']
    logger.info(f'Loading model from {model_path}...')
    utils.load_checkpoint(model_path, model)
    logger.info(f"Sending the model to '{config['device']}'")
    model = model.to(config['device'])

    logger.info('Loading test files...')
    for test_loader in get_test_loaders(config):
        logger.info(f"Processing '{test_loader.dataset.id}'...")
        
        output_file = _get_output_file(config['output_folder'], test_loader.dataset.id, format='h5')

        predictor = _get_predictor(model, test_loader, output_file, config)

        # run the model prediction on the entire dataset and save to the 'output_file'
        predictor.predict()

if __name__ == '__main__':
    main()
