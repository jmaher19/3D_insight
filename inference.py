## this file is used to run inference on shapenet renderings in order to generate
## training data for 3D-RecGAN

import os
import sys
if (sys.version_info < (3, 0)):
    raise Exception("Please follow the installation instruction on 'https://github.com/chrischoy/3D-R2N2'")

import shutil
import numpy as np
from subprocess import call

from PIL import Image
from models import load_model
from lib.config import cfg, cfg_from_list
from lib.solver import Solver
from lib.voxel import voxel2obj
import pickle
DEFAULT_WEIGHTS = 'output/ResidualGRUNet/default_model/weights.npy'


def cmd_exists(cmd):
    return shutil.which(cmd) is not None


def download_model(fn):
    if not os.path.isfile(fn):
        # Download the file if doewn't exist
        print('Downloading a pretrained model')
        call(['curl', 'ftp://cs.stanford.edu/cs/cvgl/ResidualGRUNet.npy',
              '--create-dirs', '-o', fn])

def get_input_data():
    with open('trainingdat.pickle', 'rb') as handle:
        input_data = pickle.load(handle)
        for key, value in input_data.items():
            for img in value:
                img.transpose(2,0,1).astype(np.float32)/255
            value = np.array(value)
    return input_data




def main():
    '''Main inference function'''
    # Download the input data
    input = get_input_data()

    # Download and load pretrained weights
    download_model(DEFAULT_WEIGHTS)

    # Use the default network model
    NetClass = load_model('ResidualGRUNet')

    # Define a network and a solver. Solver provides a wrapper for the test function.
    net = NetClass(compute_grad=False)  # instantiate a network
    net.load(DEFAULT_WEIGHTS)                        # load downloaded weights
    solver = Solver(net)                # instantiate a solver


    for key, value in input.items():
        voxel_prediction, _ = solver.test_output(value[0:3])
        np.save(key, voxel_prediction)


    # Save the prediction to an OBJ file (mesh file).
#    voxel2obj(pred_file_name, voxel_prediction[0, :, 1, :, :] > cfg.TEST.VOXEL_THRESH)




if __name__ == '__main__':
    # Set the batch size to 1
    cfg_from_list(['CONST.BATCH_SIZE', 1])
    main()
