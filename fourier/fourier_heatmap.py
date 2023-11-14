from typing import List, cast

import torch
import torchvision
import random
import numpy.fft as fft

def get_index_of_normalize(transforms):
    for index, transform in enumerate(transforms):
        if isinstance(transform, torchvision.transforms.Normalize):
            return index
    return None


class AddFourierNoise:
    """
    Note: Function from https://github.com/gatheluck/FourierHeatmap/blob/master/fhmap/fourier/noise.py
    Add Fourier noise to RGB channels respectively.
    This class is able to use as same as the functions in torchvision.transforms.
    Attributes:
        basis (torch.Tensor): scaled 2D Fourier basis. In the original paper, it is reperesented by 'v*U_{i,j}'.
    """

    def __init__(self, basis: torch.Tensor):
        assert len(basis.size()) == 2
        assert basis.size(0) == basis.size(1)
        self.basis = basis

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """"""
        c, h, w = x.shape[-3:]
        assert c == 3
        assert h >= 1 or w >= 1

        fourier_noise = self.basis.unsqueeze(0).repeat(c, 1, 1)  # (c, h, w)

        # Sign of noise is chosen uniformly at random from {-1, 1} per channel.
        # In the original paper,this factor is prepresented by 'r'.
        fourier_noise[0, :, :] *= random.randrange(-1, 2, 2)
        fourier_noise[1, :, :] *= random.randrange(-1, 2, 2)
        fourier_noise[2, :, :] *= random.randrange(-1, 2, 2)

        return cast(torch.Tensor, torch.clamp(x + fourier_noise, min=0.0, max=1.0).to(torch.float))


def get_spectrum(
    height: int,
    width: int,
    idx: int,
    height_ignore_edge_size: int = 0,
    width_ignore_edge_size: int = 0,
    low_center: bool = True,

) -> torch.Tensor:
    """Return spectrum matrix of 2D Fourier basis at index idx.
    Note:
        - height_ignore_edge_size and width_ignore_edge_size are used for getting subset of spectrum.
          e.g.) In the original paper, Fourier Heat Map was created for a 63x63 low frequency region for ImageNet.
        - We generate spectrum one by one to avoid waste of memory.
          e.g.) We need to generate more than 25,000 basis for ImageNet.
    Args:
        height (int): Height of spectrum.
        width (int): Width of spectrum.
        idx (int): Index to get matrix from.
        height_ignore_edge_size (int, optional): Size of the edge to ignore about height.
        width_ignore_edge_size (int, optional): Size of the edge to ignore about width.
        low_center (bool, optional): If True, returned low frequency centered spectrum.
    Yields:
        torch.Tensor spectrum matrix of 2D fourier basis
    """
    B = height * width
    indices = torch.arange(height * width)
    if low_center:
        indices = torch.cat([indices[B // 2 :], indices[: B // 2]])

    # drop ignoring edges
    indices = indices.view(height, width)
    if height_ignore_edge_size:
        indices = indices[height_ignore_edge_size:-height_ignore_edge_size, :]
    if width_ignore_edge_size:
        indices = indices[:, :-width_ignore_edge_size]
    indices = indices.flatten()

    idx = indices[idx]
    return torch.nn.functional.one_hot(idx, num_classes=B).view(
            height, width
        ).float()


def spectrum_to_basis(
    spectrum: torch.Tensor, l2_normalize: bool = True
) -> torch.Tensor:
    """Convert spectrum matrix to Fourier basis by 2D FFT. Shape of returned basis is (H, W).
    Note:
        - Currently, only supported the case H==W. If H!=W, returned basis might be wrong.
        - In order to apply 2D FFT, axes argument of numpy.fft.irfftn should be =(-2,-1).
    Args:
        spectrum (torch.Tensor): 2D spectrum matrix. Its shape should be (H, W//2+1).
                                 Here, (H, W) represent the size of 2D Fourier basis we want to get.
        l2_normalize (bool): If True, basis is l2 normalized.
    Returns:
        torch.Tensor: 2D Fourier basis.
    """
    assert len(spectrum.size()) == 2
    H = spectrum.size(-2)  # currently, only consider the case H==W
    basis = torch.from_numpy(fft.irfftn(spectrum, s=(H, H), axes=(-2, -1)))

    if l2_normalize:
        return cast(torch.Tensor, basis / basis.norm(dim=(-2, -1))[None, None])
    else:
        return cast(torch.Tensor, basis)


def create_fourier_heatmap_from_error_matrix(
    error_matrix: torch.Tensor,
) -> torch.Tensor:
    """Create Fourier Heat Map from error matrix (about quadrant 1 and 4).
    Note:
        Fourier Heat Map is symmetric about the origin.
        So by performing an inversion operation about the origin, Fourier Heat Map is created from error matrix.
    Args:
        error_matrix (torch.Tensor): The size of error matrix should be (H, H/2+1). Here, H is height of image.
                                     This error matrix shoud be about quadrant 1 and 4.
    Returns:
        torch.Tensor (torch.Tensor): Fourier Heat Map created from error matrix.
    """
    assert len(error_matrix.size()) == 2
    assert error_matrix.size(0) == 2 * (error_matrix.size(1) - 1)

    fhmap_rightside = error_matrix[1:, :-1]
    fhmap_leftside = torch.flip(fhmap_rightside, (0, 1))
    return torch.cat([fhmap_leftside[:, :-1], fhmap_rightside], dim=1)


def insert_fourier_noise(transforms: List, basis: torch.Tensor) -> None:
    """Insert Fourier noise transform to given a list of transform by inplace operation.
    Note:
        If Normalize transform is included in the given list, Fourier noise transform is added to just before Normalize transform.
        If not, Fourier noise transform is added at the end of the list.
    Args:
        transforms (List): A list of transform.
        basis (torch.Tensor): 2D Fourier basis.
    """
    normalize_index = get_index_of_normalize(transforms)
    if normalize_index is not None:
        transforms.insert(normalize_index, AddFourierNoise(basis))
    else:
        transforms.append(AddFourierNoise(basis))


