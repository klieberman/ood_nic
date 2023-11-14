import torchvision
from PIL import Image
import numpy as np

NAME = "Kodak"
IMAGE_SIZE = (512, 768)
N_IMAGES = 24
N_IMAGES_PER_CLASS = None
N_BATCHES = 1
FILE_SIZE = 24
assert N_BATCHES * FILE_SIZE == N_IMAGES, 'invalid specifications for N_IMAGES, N_BATCHES, FILE_SIZE.'

CLEAN_IMG_PATH = 'Kodak'
CORRUPT_IMG_PATH = 'Kodak-C'

NORMALIZE_TRANSFORM = torchvision.transforms.Normalize(
    mean=[0.46581238, 0.43006341, 0.35991835], std=[0.21588426, 0.22908859, 0.22003406]
)
INV_NORMALIZE_TRANSFORM = torchvision.transforms.Compose([
    torchvision.transforms.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.21588426, 1/0.22908859, 1/0.22003406]),
    torchvision.transforms.Normalize(mean = [-0.46581238, -0.43006341, -0.35991835], std = [ 1., 1., 1. ]),
                               ])

TO_TENSOR_TRANSFORM = torchvision.transforms.ToTensor()


'''
Transpose Kodak images where needed so that all images have the same size.
'''
def transpose_if_needed(img):
        if img.size == (512, 768):
            img = Image.fromarray(np.array(img).transpose(1,0,2))
        return img

CLEAN_TRANSFORM = torchvision.transforms.Compose([
    transpose_if_needed,
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