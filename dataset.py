import torch
from torchvision import datasets, transforms as T

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIClassificationIterator

import numpy as np


def get_preproc(input_size):
    return T.Compose([
        T.RandomResizedCrop(input_size, interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(input_size),
        lambda image: image.convert("RGB"),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


def dali_pipeline(batch_size, num_threads, device_id, image_dir, input_size):
    pipe = Pipeline(batch_size, num_threads, device_id)
    with pipe:
        jpegs, _ = fn.readers.file(file_root=image_dir, random_shuffle=True)
        images = fn.decoders.image(jpegs, device='mixed', output_type=types.RGB)
        images = fn.resize(images, device='gpu', resize_shorter=input_size, interp_type=types.INTERP_TRIANGULAR)
        mirror = fn.random.coin_flip(probability=0.5)
        images = fn.crop_mirror_normalize(images,
                                          dtype=types.FLOAT,
                                          output_layout="CHW",
                                          crop=(input_size, input_size),
                                          mean=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                                          std=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                                          mirror=mirror)
        pipe.set_outputs(images)
    return pipe
