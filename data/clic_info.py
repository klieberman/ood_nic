import torchvision
from PIL import Image
import numpy as np

NAME = "CLIC"
IMAGE_SIZE = (512, 768)
N_IMAGES = 428
N_IMAGES_PER_CLASS = None
N_BATCHES = 1
FILE_SIZE = 428
assert N_BATCHES * FILE_SIZE == N_IMAGES, 'invalid specifications for N_IMAGES, N_BATCHES, FILE_SIZE.'

CLEAN_IMG_PATH = 'CLIC/{}/test'
CORRUPT_IMG_PATH = 'CLIC-C/{}/test'

NORMALIZE_TRANSFORM = torchvision.transforms.Normalize(
    mean=[0.48456091, 0.4452615,  0.41254892], std=[0.28845296, 0.27559498, 0.28540007]
)
INV_NORMALIZE_TRANSFORM = torchvision.transforms.Compose([
    torchvision.transforms.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.28845296, 1/0.27559498, 1/0.28540007]),
    torchvision.transforms.Normalize(mean = [-0.48456091, -0.4452615,  -0.41254892], std = [ 1., 1., 1. ]),
                               ])

TO_TENSOR_TRANSFORM = torchvision.transforms.ToTensor()

'''
Transpose images where needed so that all images have the same size.
'''
def transpose_if_needed(img):
        if img.size == (540, 960):
            img = Image.fromarray(np.array(img).transpose(1,0,2))
        return img

CLEAN_TRANSFORM = torchvision.transforms.Compose([
    transpose_if_needed,
    torchvision.transforms.CenterCrop(IMAGE_SIZE),
    torchvision.transforms.ToTensor(),
    ])
CORRUPT_TRANSFORM = CLEAN_TRANSFORM

FHM_IMAGE_SIZE= (512, 512) # square image

# Same as clean transform except image is cropped to square
FHM_CLEAN_TRANSFORM = torchvision.transforms.Compose([
    transpose_if_needed,
    torchvision.transforms.CenterCrop(FHM_IMAGE_SIZE),
    torchvision.transforms.ToTensor(),
    ])
FHM_CLEAN_TRANSFORM_TRAIN = torchvision.transforms.Compose([
    transpose_if_needed,
    torchvision.transforms.RandomCrop(FHM_IMAGE_SIZE, pad_if_needed=True),
    torchvision.transforms.ToTensor(),
    ])