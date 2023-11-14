import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os.path as osp
import json
import seaborn as sns
import torch

from fourier.fourier_heatmap import create_fourier_heatmap_from_error_matrix


FHM_SIZE = (64, 33)


def plot_psd(
        psd_arr_file,
        size,
        save_file,
        scale=None, 
        zero_center=True, 
        cbar=True, 
        ):
    '''
    Plots the PSD arrays generated by test.py.
    @param psd_arr_file: the file with the PSD array
    @param size: the size of the PSD. Should be the dataset's image size in a 2-tuple.
    @param save_file: the file to save the figure to.
    @param scale: an optional parameter specifying the minimum and maximum value of
                    the color bar. Should be list of length two.
    @param zero_center: boolean specifying whether to set the center point of the 
                        PSD to zero. By default this is True because this value is
                        abnormally large and can distort the color bar scale for 
                        the rest of the plot. 
    @param cbar: boolean specifying whether to include the color bar.
    '''
    fig = plt.figure(dpi=300)
    psd_arr = np.load(psd_arr_file)
    center_x, center_y = psd_arr.shape[0] // 2, psd_arr.shape[1] // 2
    if zero_center:
        psd_arr[center_x, center_y] = 0.
    dx, dy = size[0] // 2, size[1] // 2
    psd_arr_valid = psd_arr[center_x - dx: center_x + dx, center_y - dy: center_y + dy]
    if scale is None:
        ticks, vmin, vmax = None, None, None
    else:
        ticks, vmin, vmax = scale, scale[0], scale[-1]
    sns.heatmap(psd_arr_valid,
            cmap="jet",
            cbar=cbar,
            cbar_kws={"ticks":ticks},
            vmin=vmin,
            vmax=vmax,
            xticklabels=False,
            yticklabels=False,
            square=True
            )
    fig.savefig(save_file)
    return


def plot_fhm(base_file_name, 
             save_file,
             cbar=True,
             metric='psnr',
             scale=None):
    error_matrix = np.zeros(FHM_SIZE)
    # Read metric from all fourier corruptions
    for r in range(FHM_SIZE[0]):
        for c in range(FHM_SIZE[1]):
            idx = r * FHM_SIZE[1] + c
            file_name = base_file_name.format(idx)
            try:
                result_dict = json.load(open(file_name))
                error_matrix[r, c] = result_dict[metric]
            except:
                print(f'File {file_name} not found.')

    error_matrix = torch.Tensor(error_matrix)
    fhm = create_fourier_heatmap_from_error_matrix(error_matrix)

    if scale is None:
        ticks, vmin, vmax = None, None, None
    else:
        ticks, vmin, vmax = scale, scale[0], scale[-1]
    fig = plt.figure(dpi=300)
    sns.heatmap(
        fhm.numpy(),
        cmap="jet",
        cbar=cbar,
        cbar_kws={"ticks":ticks},
        vmin=vmin,
        vmax=vmax,
        xticklabels=False,
        yticklabels=False,
        square=True,
    )
    fig.savefig(save_file)
    return