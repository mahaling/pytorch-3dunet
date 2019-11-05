import collections
import cloudvolume
import importlib
import numpy as np
import pickle
import torch

from torch.utils.data import Dataset, DataLoader, ConcatDataset

import augment.transforms as transforms
from datasets.hdf5 import SliceBuilder
from unet3d.utils import get_logger

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

logger = get_logger('CloudVolumeDataset')


class CloudVolumeDataset(Dataset):
    """
    Implementation of torch.utils.data.Dataset backed by downloading data from cloudvolume.
    Image and segmentation cutouts are extracted using a pickle file containing the 
    cutout bounds, id (a zero represents DO NOT USE), cloudvolume paths for image and seg
    """

    def __init__(self, image_cv, seg_cv, id, bounds, mip_level,
                 phase, patch_shape, stride_shape,
                 transformer_config, slice_builder_cls=SliceBuilder,
                 mirror_padding=False, pad_width=20):

        assert phase in ['train', 'val', 'test']
        self.phase = phase
        self._check_patch_shape(patch_shape)
        self.image_cv = image_cv
        self.seg_cv = seg_cv
        self.id = id
        assert isinstance(bounds, list)
        self.bounds = bounds
        self.mip_level = mip_level

        self.mirror_padding = mirror_padding
        self.pad_width = pad_width

        minx, maxx, miny, maxy, minz, maxz = self.bounds
        self.raws = []
        img = np.squeeze(self.image_cv[minx:maxx, miny:maxy, minz:maxz, 0])
        #img = np.transpose(img, (2, 0, 1))
        self.raws.append(img)

        mean, std = self._calculate_mean_std(self.raws[0])

        self.transformer = transforms.get_transformer(transformer_config, mean, std, phase)
        self.raw_transform = self.transformer.raw_transform()

        if phase != 'test':
            # create label/weight transform only in train/val phase
            self.label_transform = self.transformer.label_transform()
            label = np.squeeze(self.seg_cv[minx:maxx, miny:maxy, minz:maxz, 0])
            label = np.where(label/np.ndarray.max(label)>=0.2, 1, 0)
            #label = np.transpose(label, (2, 0, 1))
            self.labels = [label]

            self.weight_maps = None

            self._check_dimensionality(self.raws, self.labels)
        else:
            self.labels = None
            self.weight_maps = None

            # add mirror padding if needed
            if self.mirror_padding:
                padded_volumes = [np.pad(raw, pad_width=self.pad_width, mode='reflect') for raw in self.raws]
                self.raws = padded_volumes
        
        #print(self.raws[0].shape, self.labels[0].shape)
        #print(np.min(self.labels[0][:,:,200]), np.max(self.labels[0][:,:,200]))
        #plt.imshow(self.labels[0][:,:,200])
        #plt.show()
        #plt.imshow(self.raws[0][:,:,200])
        #plt.show()
        
        # build slice indices for raw and label data sets
        slice_builder = slice_builder_cls(self.raws, self.labels, self.weight_maps, patch_shape, stride_shape)
        self.raw_slices = slice_builder.raw_slices
        self.label_slices = slice_builder.label_slices
        self.weight_slices = slice_builder.weight_slices

        self.patch_count = len(self.raw_slices)
        logger.info(f'Number of patches: {self.patch_count}')

    def __getitem__(self, idx):
        if idx >= len(self):
            raise StopIteration

        # get the slice for a given index 'idx'
        raw_idx = self.raw_slices[idx]
        # get the raw data patch for a given slice
        raw_patch_transformed = self._transform_patches(self.raws, raw_idx, self.raw_transform)

        if self.phase == 'test':
            # just return the transformed raw patch and the metadata
            return raw_patch_transformed, raw_idx
        else:
            # get the slice for a given index 'idx'
            label_idx = self.label_slices[idx]
            label_patch_transformed = self._transform_patches(self.labels, label_idx, self.label_transform)
            if self.weight_maps is not None:
                weight_idx = self.weight_slices[idx]
                # return the transformed weight map for a given patch together with raw and label data
                weight_patch_transformed = self._transform_patches(self.weight_maps, weight_idx, self.weight_transform)
                return raw_patch_transformed, label_patch_transformed, weight_patch_transformed
            # return the transformed raw and label patches
            return raw_patch_transformed, label_patch_transformed

    def __getid__(self):
        return self.id

    @staticmethod
    def _transform_patches(datasets, label_idx, transformer):
        transformed_patches = []
        for dataset in datasets:
            # get the label data and apply the label transformer
            transformed_patch = transformer(dataset[label_idx])
            transformed_patches.append(transformed_patch)
        
        # if transformed_patches is a singleton list return the first element only
        if len(transformed_patches) == 1:
            return transformed_patches[0]
        else:
            return transformed_patches
    
    def __len__(self):
        return self.patch_count

    @staticmethod
    def _calculate_mean_std(input):
        """
        Compute a mean/std of the raw stack for normalization.
        This is an in-memory implementation, override this method
        with the chunk-based computation if you're working with huge H5 files.
        :return: a tuple of (mean, std) of the raw data
        """
        return input.mean(keepdims=True), input.std(keepdims=True)

    @staticmethod
    def _check_dimensionality(raws, labels):
        for raw in raws:
            assert raw.ndim in [3, 4], 'Raw dataset must be 3D (DxHxW) or 4D (CxDxHxW)'
            if raw.ndim == 3:
                raw_shape = raw.shape
            else:
                raw_shape = raw.shape[1:]

        for label in labels:
            assert label.ndim in [3, 4], 'Label dataset must be 3D (DxHxW) or 4D (CxDxHxW)'
            if label.ndim == 3:
                label_shape = label.shape
            else:
                label_shape = label.shape[1:]
            assert raw_shape == label_shape, 'Raw and labels have to be of the same size'

    @staticmethod
    def _check_patch_shape(patch_shape):
        assert len(patch_shape) == 3, 'patch_shape must be a 3D tuple'
        assert patch_shape[1] >= 64 and patch_shape[2] >= 64, 'Height and Width must be greater or equal 64'
        assert patch_shape[0] >= 16, 'Depth must be greater or equal 16'


def _get_slice_builder_cls(class_name):
    m = importlib.import_module('datasets.hdf5')
    clazz = getattr(m, class_name)
    return clazz

def get_train_loaders(config):
    """
    Returns dictionary containing the training and validation loaders

    :param config: a top level configuration object containing the 'loaders' key
    :return: dict {
        'train': <train_loader>
        'val': <val_loader>
    }
    """

    assert 'loaders' in config, 'Could not find data loaders configuration'
    loaders_config = config['loaders']

    logger.info('Creating training and validation set loaders...')

    # get image cloudvolume path and segmentation mask cv path
    image_cv_path = loaders_config['image_cv_path']
    seg_cv_path = loaders_config['seg_cv_path']
    cutout_pkl_file = loaders_config['cutout_pkl_file']
    mip_level = loaders_config['mip_level']

    # get train/validation patch size and stride
    train_patch = tuple(loaders_config['train_patch'])
    train_stride = tuple(loaders_config['train_stride'])
    val_patch = tuple(loaders_config['val_patch'])
    val_stride = tuple(loaders_config['val_stride'])

    # get train slice_builder_cls
    train_slice_builder_str = loaders_config.get('train_slice_builder', 'SliceBuilder')
    logger.info(f'Train slice builder class: {train_slice_builder_str}')
    train_slice_builder_cls = _get_slice_builder_cls(train_slice_builder_str)

    image_cv = cloudvolume.CloudVolume(image_cv_path, mip=mip_level, bounded=False, 
                            autocrop=True, fill_missing=True)
    seg_cv = cloudvolume.CloudVolume(seg_cv_path, mip=mip_level, bounded=False, 
                            autocrop=True, fill_missing=True)

    df = []
    with open(cutout_pkl_file, 'rb') as f:
        df = pickle.load(f)
    
    train_datasets = []
    val_datasets = []

    val_slice_builder_str = loaders_config.get('val_slice_builder', 'SliceBuilder')
    logger.info(f'Val slice builder class: {val_slice_builder_str}')
    val_slice_builder = _get_slice_builder_cls(val_slice_builder_str)

    for indx, row in df.iterrows():
        if row['id'] == 0 or row['phase'] =='test':
            continue
        row['cutout_bounds'] = list(row['cutout_bounds'])
        if row['phase'] == 'train':
            try:
                logger.info("Loading image and segmentation mask for {}".format(row['id']))
                train_dataset = CloudVolumeDataset(image_cv, seg_cv, row['id'], row['cutout_bounds'], mip_level, 'train', train_patch, train_stride, transformer_config=loaders_config['transformer'], slice_builder_cls=train_slice_builder_cls)
                train_datasets.append(train_dataset)
            except Exception:
                logger.info("Skipping training data for: {}".format(row['id']), exc_info=True)
        if row['phase'] == 'val':
            try:
                logger.info(f"Loading image and segmentation mask for {row['id']}")
                val_dataset = CloudVolumeDataset(image_cv, seg_cv, row['id'], row['cutout_bounds'],
                                                 mip_level, 'val', val_patch, val_stride,
                                                 transformer_config=loaders_config['transformer'], slice_builder_cls=val_slice_builder)
                val_datasets.append(val_dataset)
            except Exception:
                logger.info(f"Skipping validation data for: {row['id']}", exc_info=True)

    num_workers = loaders_config.get('num_workers', 1)
    logger.info(f'Number of workers for train/val dataloader: {num_workers}')
    batch_size = loaders_config.get('batch_size', 1)
    logger.info(f'Batch size for train/val loader: {batch_size}')

    # when training with volumetric data use batch_size of 1 due to GPU memory constraints
    return {
        'train': DataLoader(ConcatDataset(train_datasets), batch_size=batch_size, shuffle=True, num_workers=num_workers),
        'val': DataLoader(ConcatDataset(val_datasets), batch_size=batch_size, shuffle=True, num_workers=num_workers)
    }

def prediction_collate(batch):
    error_msg = "batch must contain tensors or slice; found {}"
    if isinstance(batch[0], torch.Tensor):
        return torch.stack(batch, 0)
    elif isinstance(batch[0], tuple) and isinstance(batch[0][0], slice):
        return batch
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [prediction_collate(samples) for samples in transposed]

    raise TypeError((error_msg.format(type(batch[0]))))


def get_test_loaders(config):
    """
    Returns a list of DataLoader, one per each test file.

    :param config: a top level configuration object containing the 'datasets' key
    :return: generator of DataLoader objects
    """

    assert 'datasets' in config, 'Could not find data sets configuration'
    datasets_config = config['datasets']

    # get test data information
    image_cv_path = datasets_config['image_cv_path']
    seg_cv_path = datasets_config['seg_cv_path']
    cutout_pkl_file = datasets_config['cutout_pkl_file']
    mip_level = datasets_config['mip_level']

    image_cv = cloudvolume.CloudVolume(image_cv_path, mip=mip_level, bounded=False, autocrop=True, fill_missing=True)
    seg_cv = cloudvolume.CloudVolume(seg_cv_path, mip=mip_level, bounded=False, autocrop=True, fill_missing=True)

    patch = tuple(datasets_config['patch'])
    stride = tuple(datasets_config['stride'])

    mirror_padding = datasets_config.get('mirror_padding', False)
    pad_width = datasets_config.get('pad_width', 20)

    if mirror_padding:
        logger.info(f'Using mirror padding. Pad width: {pad_width}')

    num_workers = datasets_config.get('num_workers', 1)
    logger.info(f'Number of workers for the dataloader: {num_workers}')

    batch_size = datasets_config.get('batch_size', 1)
    logger.info(f'Batch size for dataloader: {batch_size}')

    with open(cutout_pkl_file, 'rb') as f:
        df = pickle.load(f)

    test_datasets = []

    for indx, row in df.iterrows():
        if row['id'] == 0 or row['phase'] != 'test':
            continue
        test_dataset = CloudVolumeDataset(image_cv, seg_cv, row['id'], list(row['cutout_bounds']), 
                                          mip_level, 'test', patch, stride, 
                                          transformer_config=datasets_config['transformer'],
                                          mirror_padding=mirror_padding, pad_width=pad_width)
        test_datasets.append(test_dataset)
    
    # use generator in order to create data loaders lazily one by one
    for indx, dataset in enumerate(test_datasets):
        logger.info(f'Loading test set no: {indx}')
        yield DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=prediction_collate)