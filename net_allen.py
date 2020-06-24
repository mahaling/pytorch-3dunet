from unet3d import model
import torch
from unet3d import utils
from unet3d import config
from unet3d.model import get_model
import numpy as np

import os

in_channels = 1
out_channels = 2

final_sigmoid = False
config_file_path = os.environ['UNET_CONFIG_PATH']

config = config._load_config_yaml(config_file_path)
InstantiatedModel = get_model(config)

# InstantiatedModel = model.UNet3D( in_channels,
#                                         out_channels, 
#                                         final_sigmoid,
#                                         f_maps=32,
#                                         layer_order='crg',
#                                         num_groups=8)

InstantiatedModel.training = False

def pre_process(input_numpy_patch):
    input_numpy_patch *= 255 # chunkflow scales integer values to [0,1]
    input_numpy_patch = (input_numpy_patch - 124.7)/54.5
    #img = np.squeeze(input_numpy_patch)
    # transpose (the data is always CDHW)
    # img = np.transpose(input_numpy_patch, (0,1,4,2,3))
    print(input_numpy_patch.shape, np.max(input_numpy_patch), np.min(input_numpy_patch))
    input_patch = torch.from_numpy(input_numpy_patch)

    input_patch = input_patch.cuda()
    return input_patch

def post_process(net_output):
    # the network output did not have sigmoid applied
    print(net_output.shape)
    net_output=net_output[:,1:,:,:,:]
    print(net_output.shape)

    return net_output

def load_model(checkpointpath):
    model = InstantiatedModel
    state = utils.load_checkpoint(checkpointpath, model, map_location='cuda:0')
    model.cuda()

    model.eval()
    return model

