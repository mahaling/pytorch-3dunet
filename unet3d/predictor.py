import h5py
import hdbscan
import numpy as np
import time
import torch
import tifffile
from sklearn.cluster import MeanShift
from warnings import warn

from datasets.hdf5 import SliceBuilder
import augment.transforms as transforms
from unet3d.utils import get_logger
from unet3d.utils import unpad
from lib.chunk import Chunk
from lib.save import SaveOperator
from typing import Union

logger = get_logger('UNet3DTrainer')


class _AbstractPredictor:
    def __init__(self, model, loader, output_file, config, **kwargs):
        self.model = model
        self.loader = loader
        self.output_file = output_file
        self.config = config
        self.predictor_config = kwargs
        self.logfile = self.config.get('logfile', None)
        self.logger = get_logger('UNet3DTrainer', logfile=self.logfile)

    @staticmethod
    def _volume_shape(dataset):
        # TODO: support multiple internal datasets
        raw = dataset.raws[0]
        if raw.ndim == 3:
            return raw.shape
        else:
            return raw.shape[1:]

    @staticmethod
    def _get_output_dataset_names(number_of_datasets, prefix='predictions'):
        if number_of_datasets == 1:
            return [prefix]
        else:
            return [f'{prefix}{i}' for i in range(number_of_datasets)]

    def predict(self):
        raise NotImplementedError


class PatchInferencer:
    def __init__(self, input_patch_size: tuple, output_patch_size: tuple, 
                 output_patch_overlap: tuple, num_output_channels: int,
                 dtype: str='float32'):
        
        if output_patch_size is None:
            output_patch_size = input_patch_size
        
        self.input_patch_size = input_patch_size
        self.output_patch_size = output_patch_size
        self.output_patch_overlap = output_patch_overlap
        self.num_output_channels = num_output_channels
        
        assert len(output_patch_overlap) == 3
        assert len(input_patch_size) == 3
        assert len(output_patch_size) == 3

        self.output_offset = tuple((osz-isz)//2 for osz, isz in 
                                                   zip(input_patch_size, output_patch_size))
        
        self.input_patch_overlap = tuple((opo + 2 * ocms) for opo, ocms in 
                                         zip(output_patch_overlap, 
                                             self.output_offset))

        self.input_patch_stride = tuple(p - o for p, o in
                                        zip(input_patch_size, self.input_patch_overlap))
        self.output_patch_stride = tuple(p - o for p, o in 
                                         zip(output_patch_size, self.output_patch_overlap))

        # prepare patch mask
        self.output_patch_mask = PatchMask(output_patch_size, 
                                           output_patch_overlap,
                                           dtype=dtype)
        # keep a version in cpu for making chunk mask
        self.output_patch_mask_numpy = self.output_patch_mask

    def __call__(self, input_patch: np.ndarray) -> np.ndarray:
        r"""This method should be inherited for real implementation
        Args:
            patch: a image patch with datatype of float32,
                The value range should be in [0,1]
        
        Returns
        --------
        np.ndarray
        """
        return NotImplementedError('this function should be overload by inherited class!')

    def _reshape_patch_to_5d(self, input_patch):
        """patch should be a 5d np array
        """
        assert isinstance(input_patch, np.ndarray)
        if input_patch.ndim == 3:
            input_patch = input_patch.reshape((1, 1) + input_patch.shape)
        elif input_patch.ndim == 4:
            input_patch = input_patch.reshape((1, ) + input_patch.shape)
        return input_patch

    def _crop_output_patch(self, output_patch):
        return output_patch[:, :self.num_output_channels,
                            self.output_offset[0]:output_patch.shape[-3]-self.output_offset[0],
                            self.output_offset[1]:output_patch.shape[-2]-self.output_offset[1],
                            self.output_offset[2]:output_patch.shape[-1]-self.output_offset[2]]

class ChunkPredictor:
    def __init__(self, convnet_model, 
                 input_patch_size: Union[tuple, list],
                 output_patch_size: Union[tuple, list] = None,
                 patch_num: Union[tuple, list] = None,
                 num_output_channels: int = 1,
                 output_patch_overlap: Union[tuple, list] = (4, 64, 64),
                 output_crop_margin: Union[tuple, list] = None,
                 dtype = 'float32',
                 framework: str = 'identity',
                 batch_size: int = 1,
                 bump: str = 'wu',
                 input_size: tuple = None,
                 mask_output_chunk: bool = False,
                 mask_myelin_threshold = None,
                 dry_run: bool = False,
                 verbose: int = 1):
        
        assert input_size is None or patch_num is None

        if output_patch_size is None:
            output_patch_size = input_patch_size 
        
        self.input_patch_size = input_patch_size
        self.output_patch_size = output_patch_size
        self.output_patch_overlap = output_patch_overlap
        self.patch_num = patch_num
        self.batch_size = batch_size
        self.input_size = input_size
        self.model = convnet_model
        
        if mask_output_chunk:
            # the chunk mask will handle the boundaries 
            self.output_crop_margin = (0, 0, 0)
        else:
            if output_crop_margin is None:
                self.output_crop_margin = self.output_patch_overlap
            else:
                self.output_crop_margin = output_crop_margin
                # we should always crop more than the patch overlap 
                # since the overlap region is reweighted by patch mask
                # To-Do: equal should also be OK
                assert np.alltrue([v<=m for v, m in zip(
                    self.output_patch_overlap, 
                    self.output_crop_margin)])

        self.output_patch_crop_margin = tuple((ips-ops)//2 for ips, ops in zip(
            input_patch_size, output_patch_size))
        
        self.output_offset = tuple(opcm+ocm for opcm, ocm in zip(
            self.output_patch_crop_margin, self.output_crop_margin))
    
        self.output_patch_stride = tuple(s - o for s, o in zip(
            output_patch_size, output_patch_overlap))

        self.input_patch_overlap = tuple(opcm*2+oo for opcm, oo in zip(
            self.output_patch_crop_margin, self.output_patch_overlap))

        self.input_patch_stride = tuple(ps - po for ps, po in zip(
            input_patch_size, self.input_patch_overlap))
        
        self.patch_inference_output_offset = tuple((osz-isz)//2 for osz, isz in 
                                                   zip(self.input_patch_size, self.output_patch_size))
        
        self.patch_inference_input_patch_overlap = tuple((opo + 2 * ocms) for opo, ocms in 
                                         zip(self.output_patch_overlap, 
                                             self.patch_inference_output_offset))

        self.patch_inference_input_patch_stride = tuple(p - o for p, o in
                                        zip(self.input_patch_size, self.patch_inference_input_patch_overlap))
        self.patch_inference_output_patch_stride = tuple(p - o for p, o in 
                                         zip(self.output_patch_size, self.output_patch_overlap))


        # no chunk wise mask, the patches should be aligned inside chunk
        if not mask_output_chunk:
            assert (self.input_size is not None) or (self.patch_num is not None)
            if patch_num is None:
                assert input_size is not None
                self.patch_num = tuple((isz - o)//s for isz, o, s in zip(
                    self.input_size, self.input_patch_overlap, self.input_patch_stride))

            if self.input_size is None:
                assert self.patch_num is not None 
                self.input_size = tuple(pst*pn + po for pst, pn, po in zip(
                    self.input_patch_stride, self.patch_num, self.input_patch_overlap))
             
            self.output_size = tuple(pst*pn + po - 2*ocm for pst, pn, po, ocm in zip(
                self.output_patch_stride, self.patch_num, 
                self.output_patch_overlap, self.output_crop_margin))
        else:
            # we can handle arbitrary input and output size
            self.input_size = None 
            self.output_size = None

        self.num_output_channels = num_output_channels
        self.verbose = verbose
        self.mask_output_chunk = mask_output_chunk
        self.output_chunk_mask = None
        self.dtype = dtype        
        self.mask_myelin_threshold = mask_myelin_threshold
        self.dry_run = dry_run
        
        # allocate a buffer to avoid redundant memory allocation
        self.input_patch_buffer = np.zeros((batch_size, 1, *input_patch_size),
                                           dtype=dtype)

        self.patch_slices_list = []
        
    def _check_alignment(self):
        is_align = tuple((i - o) % s == 0 for i, s, o in zip(
            self.input_size, 
            self.patch_inference_input_patch_stride, 
            self.patch_inference_input_patch_overlap))

        # all axis should be aligned
        # the patches should aligned with input size in case
        # we will not mask the output chunk
        assert np.all(is_align)
        if self.verbose:
            print('great! patches aligns in chunk.')

    def _update_parameters_for_input_chunk(self, input_chunk):
        """
        if the input size is consistent with old one, reuse the
        patch offset list and output chunk mask. Otherwise, recompute them.
        """
        if np.array_equal(self.input_size, input_chunk.shape):
            print('reusing output chunk mask.')
            assert self.patch_slices_list is not None
        else:
            if self.input_size is not None:
                warn('the input size has changed, using new intput size.')
            self.input_size = input_chunk.shape
            
            if not self.mask_output_chunk: 
                self._check_alignment()

            self.output_size = tuple(
                isz-2*ocso for isz, ocso in 
                zip(self.input_size, self.output_offset))
        
        self.output_patch_stride = tuple(s-o for s, o in zip(
            self.output_patch_size, self.output_patch_overlap))

        self._construct_patch_slices_list(input_chunk.global_offset)

    def _construct_patch_slices_list(self, input_chunk_offset):
        """
        create the normalization mask and patch bounding box list
        """
        self.patch_slices_list = []
        # the step is the stride, so the end of aligned patch is
        # input_size - patch_overlap
        
        input_patch_size = self.input_patch_size
        output_patch_size = self.output_patch_size
        input_patch_overlap = self.input_patch_overlap 
        input_patch_stride = self.input_patch_stride 

        print('Construct patch slices list...')
        for iz in range(0, self.input_size[0] - input_patch_overlap[0], input_patch_stride[0]):
            if iz + input_patch_size[0] > self.input_size[0]:
                iz = self.input_size[0] - input_patch_size[0]
                assert iz >= 0
            iz += input_chunk_offset[-3]
            oz = iz + self.output_patch_crop_margin[0]
            for iy in range(0, self.input_size[1] - input_patch_overlap[1], input_patch_stride[1]):
                if iy + input_patch_size[1] > self.input_size[1]:
                    iy = self.input_size[1] - input_patch_size[1]
                    assert iy >= 0
                iy += input_chunk_offset[-2]
                oy = iy + self.output_patch_crop_margin[1]
                for ix in range(0, self.input_size[2] - input_patch_overlap[2], input_patch_stride[2]):
                    if ix + input_patch_size[2] > self.input_size[2]:
                        ix = self.input_size[2] - input_patch_size[2]
                        assert ix >= 0
                    ix += input_chunk_offset[-1]
                    ox = ix + self.output_patch_crop_margin[2]
                    input_patch_slice =  (slice(iz, iz + input_patch_size[0]),
                                          slice(iy, iy + input_patch_size[1]),
                                          slice(ix, ix + input_patch_size[2]))
                    output_patch_slice = (slice(oz, oz + output_patch_size[0]),
                                          slice(oy, oy + output_patch_size[1]),
                                          slice(ox, ox + output_patch_size[2]))
                    self.patch_slices_list.append((input_patch_slice, output_patch_slice))

    def _get_output_buffer(self, input_chunk):
        output_buffer_size = (self.num_output_channels, ) + self.output_size
        output_buffer_array = np.zeros(output_buffer_size, dtype=self.dtype)

        output_global_offset = tuple(io + ocso for io, ocso in zip(
            input_chunk.global_offset, self.output_offset))
        
        output_buffer = Chunk(output_buffer_array, global_offset=(0,) + output_global_offset)

        assert output_buffer == 0
        return output_buffer

    def pre_process(self, input_chunk, transformations_list = None):
        if transformations_list is None:
            return input_chunk
        else:
            mean = input_chunk.mean()
            std = input_chunk.std()
            transformer = transforms.get_transformer(transformations_list, mean, std, "test")
            raw_transform = transformer.raw_transform()
            tarray = raw_transform(input_chunk.array)
            tarray = np.squeeze(tarray.numpy(), axis=0)
            transformed_chunk = Chunk(tarray, global_offset=input_chunk.global_offset)
            return transformed_chunk

    def post_process(self, output_chunk):
        print(output_chunk.array.shape)
        net_output = output_chunk.array[0,:,:,:]
        output_chunk.array = net_output
        return output_chunk

    def predict_chunk(self, input_chunk: np.ndarray):
        """
        args:
           input_chunk (Chunk): input chunk with global offset
        """
        assert isinstance(input_chunk, Chunk)

        self._update_parameters_for_input_chunk(input_chunk)
        output_buffer = self._get_output_buffer(input_chunk)

        if not self.mask_output_chunk:
            self._check_alignment()

        if self.dry_run:
            print('dry run, return a special artificial chunk.')
            size = output_buffer.shape

            if self.mask_myelin_threshold:
                # eliminate myelin channel
                size = (size[0]-1, *size[1:])
            
            return Chunk.create(size=size, 
                                dtype=output_buffer.dtype,
                                voxel_offset=output_buffer.global_offset)
        
        if input_chunk == 0:
            print('input is all zero, return zero buffer directly')
            if self.mask_myelin_threshold:
                assert output_buffer.shape[0] == 4
                return output_buffer[:-1, ...]
            else:
                return output_buffer
        
        if np.issubdtype(input_chunk.dtype, np.integer):
            # normalize to 0-1 value range
            dtype_max = np.info(input_chunk.dtype).max
            input_chunk = input_chunk.astype(self.dtype) / dtype_max

        if self.verbose:
            chunk_time_start = time.time()

        # set model to evalutation mode
        self.model.eval()

        # send model to device
        self.model.cuda()

        with torch.no_grad():
            for i in range(0, len(self.patch_slices_list), self.batch_size):
                if self.verbose:
                    start = time.time()
                
                batch_slices = self.patch_slices_list[i:i+self.batch_size]
                for batch_idx, slices in enumerate(batch_slices):
                    self.input_patch_buffer[batch_idx, 0, :, :, :] = input_chunk.cutout(slices[0]).array

                if self.verbose > 1:
                    end = time.time()
                    print('preparing %d input patches takes %3f sec' % self.batch_size, end-start)
                    start = end

                # the input and output patch is a 5d numpy array with 
                # datatype of float32, the dimensions are (batch, channel, z, y, x)
                # the input image should be normalized to [0,1]
                patch = torch.from_numpy(self.input_patch_buffer).float().cuda()
                
                output_patch = self.model(patch)

                assert output_patch.ndim == 5

                net_out = output_patch.cpu().numpy()
                #net_out_mask = np.where(net_out >= 0.9, 1, 0)
                #print(net_out.shape)
                for batch_idx, slices in enumerate(batch_slices):
                    # slices[0] is for input patch slice
                    # slices[1] is for output patch slice
                    offset = (0,) + tuple(s.start for s in slices[1])
                    print(offset)
                    output_chunk = Chunk(net_out[batch_idx, 1:, :, :, :],
                                        global_offset=offset)
                    output_buffer.blend(output_chunk)
            
        return output_buffer

    def __call__(self, input_chunk: np.ndarray, output_volume_path, mip, pre_transforms_list = None):
        # pre process the chunk
        transformed_chunk = self.pre_process(input_chunk, pre_transforms_list)
        
        # model and chunk will be sent to gpu in predict_chunk function
        # predict the chunk
        output_chunk = self.predict_chunk(transformed_chunk)

        # post process chunk
        #output_chunk = self.post_process(output_chunk)

        # crop-margin for the output_chunk
        output_chunk.crop_margin(output_bbox = output_chunk.bbox)

        # save chunk to google cloud bucket
        saveop = SaveOperator(output_volume_path, mip)
        saveop(output_chunk)



class StandardPredictor(_AbstractPredictor):
    """
    Applies the model on the given dataset and saves the result in the `output_file` in the H5 format.
    Predictions from the network are kept in memory. If the results from the network don't fit in into RAM
    use `LazyPredictor` instead.

    The output dataset names inside the H5 is given by `des_dataset_name` config argument. If the argument is
    not present in the config 'predictions{n}' is used as a default dataset name, where `n` denotes the number
    of the output head from the network.

    Args:
        model (Unet3D): trained 3D UNet model used for prediction
        data_loader (torch.utils.data.DataLoader): input data loader
        output_file (str): path to the output H5 file
        config (dict): global config dict
    """

    def __init__(self, model, loader, output_file, config, **kwargs):
        super().__init__(model, loader, output_file, config, **kwargs)

    def predict(self):
        out_channels = self.config['model'].get('out_channels')
        if out_channels is None:
            out_channels = self.config['model']['dt_out_channels']

        prediction_channel = self.config.get('prediction_channel', None)
        if prediction_channel is not None:
            self.logger.info(f"Using only channel '{prediction_channel}' from the network output")

        device = self.config['device']
        output_heads = self.config['model'].get('output_heads', 1)

        self.logger.info(f'Running prediction on {len(self.loader)} batches...')

        # dimensionality of the the output predictions
        volume_shape = self._volume_shape(self.loader.dataset)
        if prediction_channel is None:
            prediction_maps_shape = (out_channels,) + volume_shape
        else:
            # single channel prediction map
            prediction_maps_shape = (1,) + volume_shape

        self.logger.info(f'The shape of the output prediction maps (CDHW): {prediction_maps_shape}')

        avoid_block_artifacts = self.predictor_config.get('avoid_block_artifacts', True)
        self.logger.info(f'Avoid block artifacts: {avoid_block_artifacts}')

        # create destination H5 file
        h5_output_file = h5py.File(self.output_file, 'w')
        
        
        # allocate prediction and normalization arrays
        self.logger.info('Allocating prediction and normalization arrays...')
        prediction_maps, normalization_masks = self._allocate_prediction_maps(prediction_maps_shape,
                                                                              output_heads, h5_output_file)

        # Sets the module in evaluation mode explicitly, otherwise the final Softmax/Sigmoid won't be applied!
        self.model.eval()
        # Run predictions on the entire input dataset
        with torch.no_grad():
            for batch, indices in self.loader:
                # send batch to device
                batch = batch.to(device)

                # forward pass
                predictions = self.model(batch)

                # wrap predictions into a list if there is only one output head from the network
                if output_heads == 1:
                    predictions = [predictions]

                # for each output head
                for prediction, prediction_map, normalization_mask in zip(predictions, prediction_maps,
                                                                          normalization_masks):

                    # convert to numpy array
                    prediction = prediction.cpu().numpy()

                    # for each batch sample
                    for pred, index in zip(prediction, indices):
                        # save patch index: (C,D,H,W)
                        if prediction_channel is None:
                            channel_slice = slice(0, out_channels)
                        else:
                            channel_slice = slice(0, 1)
                        index = (channel_slice,) + index

                        if prediction_channel is not None:
                            # use only the 'prediction_channel'
                            self.logger.info(f"Using channel '{prediction_channel}'...")
                            pred = np.expand_dims(pred[prediction_channel], axis=0)

                        self.logger.info(f'Saving predictions for slice:{index}...')

                        if avoid_block_artifacts:
                            # unpad in order to avoid block artifacts in the output probability maps
                            u_prediction, u_index = unpad(pred, index, volume_shape)
                            # accumulate probabilities into the output prediction array
                            prediction_map[u_index] += u_prediction
                            # count voxel visits for normalization
                            normalization_mask[u_index] += 1
                        else:
                            # accumulate probabilities into the output prediction array
                            prediction_map[index] += pred
                            # count voxel visits for normalization
                            normalization_mask[index] += 1

        # save results to
        self._save_results(prediction_maps, normalization_masks, output_heads, h5_output_file, self.loader.dataset)
        
        # close the output H5 file
        h5_output_file.close()

    def _allocate_prediction_maps(self, output_shape, output_heads, output_file):
        # initialize the output prediction arrays
        prediction_maps = [np.zeros(output_shape, dtype='float32') for _ in range(output_heads)]
        # initialize normalization mask in order to average out probabilities of overlapping patches
        normalization_masks = [np.zeros(output_shape, dtype='uint8') for _ in range(output_heads)]
        return prediction_maps, normalization_masks

    def _save_results(self, prediction_maps, normalization_masks, output_heads, output_file, dataset):
        # save probability maps
        prediction_datasets = self._get_output_dataset_names(output_heads, prefix='predictions')
        for prediction_map, normalization_mask, prediction_dataset in zip(prediction_maps, normalization_masks,
                                                                          prediction_datasets):
            prediction_map = prediction_map / normalization_mask

            if dataset.mirror_padding:
                pad_width = dataset.pad_width
                self.logger.info(f'Dataset loaded with mirror padding, pad_width: {pad_width}. Cropping before saving...')

                prediction_map = prediction_map[:, pad_width:-pad_width, pad_width:-pad_width, pad_width:-pad_width]

            self.logger.info(f'Saving predictions to: {output_file}/{prediction_dataset}...')
            output_file.create_dataset(prediction_dataset, data=prediction_map, compression="gzip")


class LazyPredictor(StandardPredictor):
    """
        Applies the model on the given dataset and saves the result in the `output_file` in the H5 format.
        Predicted patches are directly saved into the H5 and they won't be stored in memory. Since this predictor
        is slower than the `StandardPredictor` it should only be used when the predicted volume does not fit into RAM.

        The output dataset names inside the H5 is given by `des_dataset_name` config argument. If the argument is
        not present in the config 'predictions{n}' is used as a default dataset name, where `n` denotes the number
        of the output head from the network.

        Args:
            model (Unet3D): trained 3D UNet model used for prediction
            data_loader (torch.utils.data.DataLoader): input data loader
            output_file (str): path to the output H5 file
            config (dict): global config dict
        """

    def __init__(self, model, loader, output_file, config, **kwargs):
        super().__init__(model, loader, output_file, config, **kwargs)

    def _allocate_prediction_maps(self, output_shape, output_heads, output_file):
        # allocate datasets for probability maps
        prediction_datasets = self._get_output_dataset_names(output_heads, prefix='predictions')

        prediction_maps = [
            output_file.create_dataset(dataset_name, shape=output_shape, dtype='float32', chunks=True,
                                       compression='gzip')
            for dataset_name in prediction_datasets]

        # allocate datasets for normalization masks
        normalization_datasets = self._get_output_dataset_names(output_heads, prefix='normalization')
        normalization_masks = [
            output_file.create_dataset(dataset_name, shape=output_shape, dtype='uint8', chunks=True,
                                       compression='gzip')
            for dataset_name in normalization_datasets]

        return prediction_maps, normalization_masks

    def _save_results(self, prediction_maps, normalization_masks, output_heads, output_file, dataset):
        if dataset.mirror_padding:
            self.logger.warn(
                f'Mirror padding unsupported in LazyPredictor. Output predictions will be padded with pad_width: {dataset.pad_width}')

        prediction_datasets = self._get_output_dataset_names(output_heads, prefix='predictions')
        normalization_datasets = self._get_output_dataset_names(output_heads, prefix='normalization')

        # normalize the prediction_maps inside the H5
        for prediction_map, normalization_mask, prediction_dataset, normalization_dataset in zip(prediction_maps,
                                                                                                 normalization_masks,
                                                                                                 prediction_datasets,
                                                                                                 normalization_datasets):
            # split the volume into 4 parts and load each into the memory separately
            self.logger.info(f'Normalizing {prediction_dataset}...')

            z, y, x = prediction_map.shape[1:]
            # take slices which are 1/27 of the original volume
            patch_shape = (z // 3, y // 3, x // 3)
            for index in SliceBuilder._build_slices(prediction_map, patch_shape=patch_shape, stride_shape=patch_shape):
                self.logger.info(f'Normalizing slice: {index}')
                prediction_map[index] /= normalization_mask[index]
                # make sure to reset the slice that has been visited already in order to avoid 'double' normalization
                # when the patches overlap with each other
                normalization_mask[index] = 1

            self.logger.info(f'Deleting {normalization_dataset}...')
            del output_file[normalization_dataset]


class EmbeddingsPredictor(_AbstractPredictor):
    """
    Applies the embedding model on the given dataset and saves the result in the `output_file` in the H5 format.

    The resulting volume is the segmentation itself (not the embedding vectors) obtained by clustering embeddings
    with HDBSCAN or MeanShift algorithm patch by patch and then stitching the patches together.
    """

    def __init__(self, model, loader, output_file, config, clustering, iou_threshold=0.7, noise_label=-1, **kwargs):
        super().__init__(model, loader, output_file, config, **kwargs)

        self.iou_threshold = iou_threshold
        self.noise_label = noise_label
        self.clustering = clustering

        assert clustering in ['hdbscan', 'meanshift'], 'Only HDBSCAN and MeanShift are supported'
        self.logger.info(f'IoU threshold: {iou_threshold}')

        self.clustering_name = clustering
        self.clustering = self._get_clustering(clustering, kwargs)

    def predict(self):
        device = self.config['device']
        output_heads = self.config['model'].get('output_heads', 1)

        self.logger.info(f'Running prediction on {len(self.loader)} patches...')

        # dimensionality of the the output segmentation
        volume_shape = self._volume_shape(self.loader.dataset)

        self.logger.info(f'The shape of the output segmentation (DHW): {volume_shape}')

        self.logger.info('Allocating segmentation array...')
        # initialize the output prediction arrays
        output_segmentations = [np.zeros(volume_shape, dtype='int32') for _ in range(output_heads)]
        # initialize visited_voxels arrays
        visited_voxels_arrays = [np.zeros(volume_shape, dtype='uint8') for _ in range(output_heads)]

        # Sets the module in evaluation mode explicitly
        self.model.eval()
        # Run predictions on the entire input dataset
        with torch.no_grad():
            for batch, indices in self.loader:
                # self.logger.info(f'Predicting embeddings for slice:{index}')

                # send batch to device
                batch = batch.to(device)
                # forward pass
                embeddings = self.model(batch)

                # wrap predictions into a list if there is only one output head from the network
                if output_heads == 1:
                    embeddings = [embeddings]

                for prediction, output_segmentation, visited_voxels_array in zip(embeddings, output_segmentations,
                                                                                 visited_voxels_arrays):

                    # convert to numpy array
                    prediction = prediction.cpu().numpy()

                    # iterate sequentially because of the current simple stitching that we're using
                    for pred, index in zip(prediction, indices):
                        # convert embeddings to segmentation with hdbscan clustering
                        segmentation = self._embeddings_to_segmentation(pred)
                        # stitch patches
                        self._merge_segmentation(segmentation, index, output_segmentation, visited_voxels_array)

        # save results
        with h5py.File(self.output_file, 'w') as output_file:
            prediction_datasets = self._get_output_dataset_names(output_heads,
                                                                 prefix=f'segmentation/{self.clustering_name}')
            for output_segmentation, prediction_dataset in zip(output_segmentations, prediction_datasets):
                self.logger.info(f'Saving predictions to: {output_file}/{prediction_dataset}...')
                output_file.create_dataset(prediction_dataset, data=output_segmentation, compression="gzip")

    def _embeddings_to_segmentation(self, embeddings):
        """
        Cluster embeddings vectors with HDBSCAN and return the segmented volume.

        Args:
            embeddings (ndarray): 4D (CDHW) embeddings tensor
        Returns:
            3D (DHW) segmentation
        """
        # shape of the output segmentation
        output_shape = embeddings.shape[1:]
        # reshape (C, D, H, W) -> (C, D * H * W) and transpose -> (D * H * W, C)
        flattened_embeddings = embeddings.reshape(embeddings.shape[0], -1).transpose()

        self.logger.info('Clustering embeddings...')
        # perform clustering and reshape in order to get the segmentation volume
        start = time.time()
        clusters = self.clustering.fit_predict(flattened_embeddings).reshape(output_shape)
        self.logger.info(
            f'Number of clusters found by {self.clustering}: {np.max(clusters)}. Duration: {time.time() - start} sec.')
        return clusters

    def _merge_segmentation(self, segmentation, index, output_segmentation, visited_voxels_array):
        """
        Given the `segmentation` patch, its `index` in the `output_segmentation` array and the array visited voxels
        merge the segmented patch (`segmentation`) into the `output_segmentation`

        Args:
            segmentation (ndarray): segmented patch
            index (tuple): position of the patch inside `output_segmentation` volume
            output_segmentation (ndarray): current state of the output segmentation
            visited_voxels_array (ndarray): array of voxels visited so far (same size as `output_segmentation`); visited
                voxels will be marked by a number greater than 0
        """
        index = tuple(index)
        # get new unassigned label
        max_label = np.max(output_segmentation) + 1
        # make sure there are no clashes between current segmentation patch and the output_segmentation
        # but keep the noise label
        noise_mask = segmentation == self.noise_label
        segmentation += int(max_label)
        segmentation[noise_mask] = self.noise_label
        # get the overlap mask in the current patch
        overlap_mask = visited_voxels_array[index] > 0
        # get the new labels inside the overlap_mask
        new_labels = np.unique(segmentation[overlap_mask])
        merged_labels = self._merge_labels(output_segmentation[index], new_labels, segmentation)
        # relabel new segmentation with the merged labels
        for current_label, new_label in merged_labels:
            segmentation[segmentation == new_label] = current_label
        # update the output_segmentation
        output_segmentation[index] = segmentation
        # visit the patch
        visited_voxels_array[index] += 1

    def _merge_labels(self, current_segmentation, new_labels, new_segmentation):
        def _most_frequent_label(labels):
            unique, counts = np.unique(labels, return_counts=True)
            ind = np.argmax(counts)
            return unique[ind]

        result = []
        # iterate over new_labels and merge regions if the IoU exceeds a given threshold
        for new_label in new_labels:
            # skip 'noise' label assigned by hdbscan
            if new_label == self.noise_label:
                continue
            new_label_mask = new_segmentation == new_label
            # get only the most frequent overlapping label
            most_frequent_label = _most_frequent_label(current_segmentation[new_label_mask])
            # skip 'noise' label
            if most_frequent_label == self.noise_label:
                continue
            current_label_mask = current_segmentation == most_frequent_label
            # compute Jaccard index
            iou = np.bitwise_and(new_label_mask, current_label_mask).sum() / np.bitwise_or(new_label_mask,
                                                                                           current_label_mask).sum()
            if iou > self.iou_threshold:
                # merge labels
                result.append((most_frequent_label, new_label))

        return result

    def _get_clustering(self, clustering_alg, kwargs):
        self.logger.info(f'Using {clustering_alg} for clustering')

        if clustering_alg == 'hdbscan':
            min_cluster_size = kwargs.get('min_cluster_size', 50)
            min_samples = kwargs.get('min_samples', None),
            metric = kwargs.get('metric', 'euclidean')
            cluster_selection_method = kwargs.get('cluster_selection_method', 'eom')

            self.logger.info(f'HDBSCAN params: min_cluster_size: {min_cluster_size}, min_samples: {min_samples}')
            return hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric=metric,
                                   cluster_selection_method=cluster_selection_method)
        else:
            bandwidth = kwargs['bandwidth']
            self.logger.info(f'MeanShift params: bandwidth: {bandwidth}, bin_seeding: True')
            # use fast MeanShift with bin seeding
            return MeanShift(bandwidth=bandwidth, bin_seeding=True)
