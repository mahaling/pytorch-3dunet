from unet3d import model
import torch
from unet3d import utils
in_channels = 1
out_channels = 2

final_sigmoid = False

InstantiatedModel = model.UNet3D( in_channels,
                                        out_channels, 
                                        final_sigmoid,
                                        f_maps=32,
                                        layer_order='crg',
                                        num_groups=8)

InstantiatedModel.training = False

def pre_process(input_numpy_patch):
    input_numpy_patch *= 255 # chunkflow scales integer values to [0,1]
    return input_numpy_patch

def post_processing(net_output):
    # the network output did not have sigmoid applied
    output_patch = net_output.sigmoid()

    return output_patch

def load_model(checkpointpath):
    model = InstantiatedModel
    state = utils.load_checkpoint(checkpointpath, model, map_location='gpu')
    model.cuda()
    return model