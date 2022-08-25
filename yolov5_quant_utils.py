import tensorflow as tf
import torch
from PIL import Image
from bbox_utils import letterbox, to_numpy
import numpy as np
from pathlib import Path
import pathlib
from itertools import islice
import os
import glob


def datasetGenerateImagesYolov5(image_size,image_mask, maximum_match, print_filenames ):
    """
    Dataset generator for post-training quantization based on supplied search mask
    [in] image_mask - image search mask string: some/path/**/*.jpg - look in all subfolders of some/path/ for files with jpg extention
    [in] maximum_match - maximum number of files to yield
    [in] image_init - initial image transformation 
    [in] preprocessors_list - image preprocessor functions list
    """
    for filename in islice(glob.iglob(os.path.abspath(image_mask), recursive=True), maximum_match):
        if print_filenames:
            print(filename)
        img = np.asarray(Image.open(str(filename)).convert('RGB'))
        letterbox_img,ratio, pad=letterbox(img,new_shape=(image_size[0], image_size[1]),auto=False, scaleFill=False,stride=1)
        img_array = np.array(letterbox_img).astype(np.float32)
        img_array2=(img_array)/255
        torch_image=torch.tensor(img_array2)
        keras_input=torch.cat([torch_image[..., ::2, ::2,:], torch_image[..., 1::2, ::2,:], torch_image[..., ::2, 1::2,:], torch_image[..., 1::2, 1::2,:]], 2)
        y=to_numpy(keras_input)
        yield [tf.expand_dims(y, axis=0)]
        
