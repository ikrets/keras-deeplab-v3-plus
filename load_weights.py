from __future__ import print_function

import os
import numpy as np
from tqdm import tqdm
from model import Deeplabv3
import sys

MODEL_DIR = 'models'

backbone = 'mobilenetv2'
print('Instantiating an empty Deeplabv3+ model...')
model = Deeplabv3(input_shape=(512, 512, 3), alpha=1,
                  classes=21, OS=16, backbone=backbone, weights=None)

WEIGHTS_DIR = 'weights/' + backbone
print('Loading weights from', WEIGHTS_DIR)
for layer in tqdm(model.layers):
    if layer.weights:
        weights = []
        for w in layer.weights:
            weight_name = os.path.basename(w.name).replace(':0', '')
            weight_file = layer.name + '_' + weight_name + '.npy'
            weight_arr = np.load(os.path.join(WEIGHTS_DIR, weight_file))
            weights.append(weight_arr)
        try:
            layer.set_weights(weights)
        except Exception as ex:
            print('Problem with layer {}: {}'.format(layer.name, ex))
            sys.exit()

print('Saving model weights...')
OUTPUT_WEIGHT_FILENAME = 'deeplabv3_' + \
    backbone + '_tf_dim_ordering_tf_kernels.h5'
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
model.save_weights(os.path.join(MODEL_DIR, OUTPUT_WEIGHT_FILENAME))
