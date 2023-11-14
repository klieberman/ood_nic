import torch
import torch.nn.functional as F
import torchvision.transforms as T

import compressai
import os.path as osp
import os
import time
import tqdm
import json
from collections import defaultdict
from pytorch_msssim import ms_ssim
from PIL import Image
import numpy as np
from skimage.color import rgb2gray
import pandas as pd
import math
import copy
from argparse import Namespace

from tempfile import mkstemp
from models.codecs import JPEG2000, VTM, JPEG

from args.test_args import TestArguments
from data.corruptions import CORRUPTIONS
from fourier.fourier_heatmap import get_spectrum, spectrum_to_basis, insert_fourier_noise
from models.scale_hyperprior import ScaleHyperprior
from models.scale_hyperprior_lct import ScaleHyperpriorLCT
from models.elic import ELICModel
from utils.classify import get_classification_model, classify_batch
from utils.compression import psnr, scale_to_255, tensor_to_arr
from utils.elic_utils import compute_bpp_elic
from utils.model_helpers import resume
from utils.utilities import get_device, makedirs_if_needed

# ELIC requires deterministic
torch.backends.cudnn.deterministic = True
torch.set_num_threads(1)


TO_TENSOR_TRANSFORM = T.ToTensor()

'''
Script for testing neural image compression models.
Based off of: https://github.com/InterDigitalInc/CompressAI/blob/master/compressai/utils/eval_model/__main__.py
Includes functionality to
    - get rate-distortion results (bpp, mse, ms-ssim)
    - get PSD arrays
    - classify reconstructed images using pre-trained ResNet
'''

@torch.no_grad()
def nic_inference(model, x, lmbda=None, clean=None, elic=False):
    h, w = x.size(2), x.size(3)
    p = 64  # maximum 6 strides of 2
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    x_padded = F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )

    start = time.time()
    if lmbda is None:
        out_enc = model.compress(x_padded)
    else:
        out_enc = model.compress(x_padded, lmbda)
    enc_time = time.time() - start

    start = time.time()
    if lmbda is None:
        out_dec = model.decompress(out_enc["strings"], out_enc["shape"])
    else:
        out_dec = model.decompress(out_enc["strings"], out_enc["shape"], lmbda)
    dec_time = time.time() - start

    out_dec["x_hat"] = F.pad(
        out_dec["x_hat"], (-padding_left, -padding_right, -padding_top, -padding_bottom)
    )

    n_pixels = x.size(0) * x.size(2) * x.size(3)
    
    if elic:
        bpp = compute_bpp_elic(out_enc, n_pixels)
    else:
        bpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / n_pixels

    rv = {
        "psnr": psnr(x, out_dec["x_hat"]),
        "ms-ssim": ms_ssim(x, out_dec["x_hat"], data_range=1.0).item(),
        "bpp": bpp,
        "encoding_time": enc_time,
        "decoding_time": dec_time,
    }

    if clean is not None:
        rv["psnr_wrt_clean"] = psnr(clean, out_dec["x_hat"])

    return out_dec['x_hat'], rv


def inference_entropy_estimation(model, x, lmbda=None, clean=None):
    start_time = time.time()
    if lmbda is None:
        out_net = model.forward(x)
    else:
        out_net = model.forward(x, lmbda)
    elapsed_time = time.time() - start_time

    n_pixels = x.size(0) * x.size(2) * x.size(3)
    bpp = sum(
        (torch.log(likelihoods).sum() / (-math.log(2) * n_pixels))
        for likelihoods in out_net["likelihoods"].values()
    )

    rv = {
        "psnr": psnr(x, out_net["x_hat"]),
        "bpp": bpp.item(),
        "encoding_time": elapsed_time / 2.0,  # broad estimation
        "decoding_time": elapsed_time / 2.0,
    }

    if clean is not None:
        rv["psnr_wrt_clean"] = psnr(clean, out_net["x_hat"])

    return out_net['x_hat'], rv


def classic_inference(codec, img, quality, device, clean, vvc=False):

    # Save image after transform (e.g. cropping) to temporary file
    fd0, filepath = mkstemp(suffix=".png")
    save_image(img, filepath)

    # Run classic codec encoding and decoding, calculate psnr and ms-ssim
    rv, x_hat = codec.run(filepath, quality, None, device, return_rec=True)
    os.close(fd0)
    os.remove(filepath)

    # Transfer image to tensor batch for consistency with NIC inference method
    x_hat = torch.unsqueeze(TO_TENSOR_TRANSFORM(x_hat), 0).to(device)

    if clean is not None:
        rv["psnr_wrt_clean"] = psnr(clean, x_hat)
    
    return x_hat, rv


def save_image(x_hat, save_name):
    assert len(x_hat) == 1, 'can only save 1 image at a time.'
    img_tensor = scale_to_255(x_hat[0])
    img_arr = tensor_to_arr(img_tensor)
    img = Image.fromarray(img_arr, 'RGB')
    img.save(save_name)
    return


def get_power_spectrums(img, square=True):
    '''Compute 2D power spectral density of of grayscale version of image'''
    img_grey = rgb2gray(np.round(img))
    img_grey_F = np.fft.fftshift(np.fft.fft2(img_grey))
    ps2D = np.abs(img_grey_F)
    return ps2D


def add_psds(psd_tmp, psd_sum):
    if (True in np.isinf(psd_tmp)):
        print('Index is infinite')
        return psd_sum
    else:
        return psd_sum + psd_tmp


if __name__ == '__main__':
    test_args = TestArguments()
    args = test_args.parse()

    device = get_device()

    compressai.set_entropy_coder(args.entropy_coder)

    if args.no_compression:
        assert args.resume is None, "Can't resume no-compression set to True."
        assert args.classic_codec is None, "Can't have no_compression set to True and classic_codec != None."
        epoch = 0
    elif args.classic_codec is not None:
        assert args.resume is None, "Can't resume with a classic codec."
        assert args.quality is not None, 'q must be set when classic_codec != None.'
        assert args.batch_size == 1, 'expected batch size of 1 for classic_codec.'
        if args.classic_codec == "jpeg2k":
            codec = JPEG2000({})
        elif args.classic_codec == "vtm":
            codec = VTM(Namespace(build_dir=args.vtm_build_dir,
                                    config=args.vtm_config,
                                    rgb=False
                                    )
            )
        elif args.classic_codec == "jpeg":
            codec = JPEG({})
        quality = args.quality
        epoch = 0
    else:
        if args.model == "scale_hyperprior":
            if args.variable_rate:
                print(f"Training model over lambda range {args.lambda_range} \
                with {args.prune_algorithm} pruning algorithm.\n")
                
                model = ScaleHyperpriorLCT(N=args.N, M=args.M, args=args)
                
            else:                
                model = ScaleHyperprior(N=args.N, M=args.M, args=args)
                
        elif args.model == "elic":
            model = ELICModel(N=args.N, M=args.M)
        else:
            exit(f'Error: invalid model choice {args.model}')
 
        model = model.to(device)

        assert args.resume is not None and osp.exists(args.resume), f'invalid resume path {args.resume}'
        print(f"Resuming from {args.resume}.\n")
        model, _, epoch, _ = resume(args, model, None, None, confirm_sparsity=False)
        
    # Get dataset and loader
    if args.test_dataset == 'imagenet':
        import data.imagenet_info as data_info
        from torchvision.datasets import ImageFolder 
    elif args.test_dataset == 'kodak':
        import data.kodak_info as data_info
        from data.compression import ImageFolder as ImageFolder
    elif args.test_dataset == 'clic':
        import data.clic_info as data_info
        from data.compression import ImageFolderCLIC as ImageFolder
    else:
        sys.exit(f'Invalid test dataset {args.test_dataset}.')

    kwargs = {}

    # Get clean dataset
    clean_folder = osp.join(args.data_prefix, data_info.CLEAN_IMG_PATH)
    if args.fhm_idx is not None:
        clean_dataset = ImageFolder(clean_folder, data_info.FHM_CLEAN_TRANSFORM)
    else:
        clean_dataset = ImageFolder(clean_folder, data_info.CLEAN_TRANSFORM)

    n_images = len(clean_dataset)
    clean_data_str = f'{args.test_dataset}_clean'
    assert len(clean_dataset) > 0, f'clean_dataset {clean_folder} has length 0.'
    print(f"Clean dataset loaded from {clean_folder} has {n_images} \
    ({n_images/args.batch_size} batches of size {args.batch_size}).")

    # Get corrupt dataset if needed (either for Fourier heat map or -C dataset)
    if args.fhm_idx is not None:
        assert args.corruption == 0, 'Fourier heatmap corruptions only work with clean data.'
        if not isinstance(data_info.FHM_CLEAN_TRANSFORM, T.Compose):
            raise ValueError(
                f"type of dataset.transform should be torchvision.transforms.Compose, not {type(dataset.transform)}"
            )
        height = data_info.FHM_IMAGE_SIZE[0]
        width = height // 2 + 1
        ignore_edge_size = (height - args.fhm_size[0]) // 2
        spectrum = get_spectrum(height, width, args.fhm_idx, ignore_edge_size, ignore_edge_size)
        eps = 4.
        basis = spectrum_to_basis(spectrum, l2_normalize=True) * eps

        # Insert Fourier Noise transform into list of transforms and make test dataset with
        # noised_transforms
        noised_transform = copy.deepcopy(data_info.FHM_CLEAN_TRANSFORM)
        insert_fourier_noise(noised_transform.transforms, basis)
        test_dataset = ImageFolder(clean_folder, noised_transform)

        data_str = f'{args.test_dataset}_fhm_{args.fhm_idx}'
        assert n_images == len(test_dataset), 'expected n_images to be same for clean and corrupt datasets.'
        print(f"Test dataset loaded from {clean_folder} with fourier noise from index {args.fhm_idx}.")

    else:
        if args.corruption == 0:
            test_dataset = clean_dataset
            data_str = clean_data_str
        else:
            corruption = CORRUPTIONS[args.corruption]
            test_transform = data_info.CORRUPT_TRANSFORM
            test_folder = osp.join(args.data_prefix, data_info.CORRUPT_IMG_PATH, corruption, args.severity)
            data_str = f'{args.test_dataset}_{corruption}_{args.severity}'

            test_dataset = ImageFolder(test_folder, transform=test_transform)
            assert n_images == len(test_dataset), 'expected n_images to be same for clean and corrupt datasets.'
            print(f"Test dataset loaded from {test_folder}.")

    if args.subset:
        assert args.test_dataset == "imagenet", "subset is only valid for ImageNet."
        indices = np.array([])
        for i in range(5):
            indices = np.concatenate((indices, np.arange(i, 50000, 50)))
        indices = indices.astype(np.uint)
        clean_dataset = torch.utils.data.Subset(clean_dataset, indices)
        test_dataset = torch.utils.data.Subset(test_dataset, indices)
        n_images = len(clean_dataset)
        print(f'Clean and test datasets subseted. New size: {n_images}.')

    clean_loader = torch.utils.data.DataLoader(
            clean_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            **kwargs
        )
    n_batches = len(clean_loader)
    test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            **kwargs
        )


    if args.psd:
        assert args.batch_size == 1, 'expected batch size of 1 to compute PSDs.'
        psd_size = data_info.IMAGE_SIZE

        if args.psd_no_diff:
            psd_sum = np.zeros(psd_size)
        else:
            psd_diff_clean_sum = np.zeros(psd_size)
            if args.corruption != 0:
                psd_diff_corrupt_sum = np.zeros(psd_size)

    if args.classify:
        classification_model = get_classification_model(device)
        classify_df = pd.DataFrame(columns=['y', 'y_hat', 'softmax[y_hat]'])

        
    # Get relevant directories, save config file
    if args.rep_count is not None:
        run_dir = osp.join(args.run_dir, args.rep_count)
    else:
        run_dir = args.run_dir
    print(f"Saving results in {run_dir}.\n")
    config_dir = makedirs_if_needed(osp.join(run_dir, 'configs'))
    test_args.save(osp.join(config_dir, f"test_epoch_{epoch}.txt"))
    rd_dir = makedirs_if_needed(osp.join(run_dir, 'results_rd'))
    if args.fhm_idx is not None:
        rd_dir = makedirs_if_needed(osp.join(rd_dir, 'fhm'))
    if args.save_images:
        image_dir = makedirs_if_needed(osp.join(run_dir, 'results_images', data_str))
        print(f"Saving images to {image_dir}")
    if args.psd:
        psd_dir = makedirs_if_needed(osp.join(run_dir, 'results_psds'))
    if args.classify:
        assert args.test_dataset == 'imagenet', 'Classification only implemented for imagenet dataset.'
        classify_dir = makedirs_if_needed(osp.join(run_dir, 'results_classify'))
    
    if not args.no_compression:
        # Prepare to test model
        if args.variable_rate:
            assert args.eval_lambda is not None, 'args.eval_lambda must be specified if args.variable_rate is true.'
            lmbda_arr = [args.eval_lambda] * args.batch_size
            lmbda_tensor = torch.Tensor(lmbda_arr).unsqueeze(1).to(device)
            if args.elic:
                # ELIC-QVRF takes an index as lambda
                lmbda_tensor = lmbda_tensor.type(torch.long)
        else:
            lmbda_tensor = None

        if args.batch_size != 1:
            assert args.entropy_estimation, 'batch size must be 1 if args.entropy_estimation is False.'

        if args.classic_codec is None:
            model.update() # required by NIC models
            model.eval()
    
    # Initialize metrics dictionary
    metrics = defaultdict(float)

    # Test loop
    for i, (x, clean_x) in tqdm.tqdm(enumerate(zip(test_loader, clean_loader)), ascii=True, total=n_batches, disable=args.disable_tqdm):
        if args.test_dataset == 'imagenet':
            x, y = x
            clean_x, _ = clean_x
             
        batch_size = len(x)
        if args.no_compression:
            x_hat = x.to(device)
            clean_x = clean_x.to(device)
            if args.corruption != 0 or args.fhm_idx is not None:
                rv = {
                    'psnr_wrt_clean': psnr(x_hat, clean_x)
                }
            else:
                rv = None
        else:
            if args.classic_codec is not None:
                clean_x = clean_x.to(device)
                x_hat, rv = classic_inference(codec, x, quality, device, clean_x)
            else:
                x = x.to(device)
                clean_x = clean_x.to(device)
                if args.entropy_estimation:
                    if args.variable_rate:
                        lmbda_tensor_tmp = lmbda_tensor[:batch_size]
                    else:
                        lmbda_tensor_tmp = None
                    x_hat, rv = inference_entropy_estimation(model, x, lmbda_tensor_tmp, clean_x)
                else:
                    if args.half:
                        model = model.half()
                        x = x.half()
                    x_hat, rv = nic_inference(model, x, lmbda_tensor, clean_x, args.model == 'elic')

        if rv is not None:
            for k, v in rv.items():
                # print(k, v)
                if not math.isinf(v):
                    metrics[k] += v
                else:
                    print(f"Index {i} {k} has value {v}.")

        if args.save_images:
            save_image(clean_x, osp.join(image_dir, f'{i}_clean.png'))
            save_image(x, osp.join(image_dir, f'{i}.png'))
            save_image(x_hat, osp.join(image_dir, f'{i}_hat.png'))

        if args.psd:
            x_hat_norm = data_info.NORMALIZE_TRANSFORM(x_hat)
            x_hat_norm_img = tensor_to_arr(x_hat_norm[0].detach())
            if args.psd_no_diff:
                psd_tmp = get_power_spectrums(x_hat_norm_img)
                psd_sum = add_psds(psd_tmp, psd_sum)
            else:
                clean_norm = data_info.NORMALIZE_TRANSFORM(clean_x)
                clean_norm_img = tensor_to_arr(clean_norm[0].detach())
                psd_diff_clean = get_power_spectrums(x_hat_norm_img - clean_norm_img)
                psd_diff_clean_sum = add_psds(psd_diff_clean, psd_diff_clean_sum)
                if args.corruption != 0:
                    corrupt_norm = data_info.NORMALIZE_TRANSFORM(x)
                    corrupt_norm_img = tensor_to_arr(corrupt_norm[0].detach())
                    psd_diff_corrupt = get_power_spectrums(x_hat_norm_img - corrupt_norm_img)
                    psd_diff_corrupt_sum = add_psds(psd_diff_corrupt, psd_diff_corrupt_sum)


        if args.classify:
            x_hat_norm = data_info.NORMALIZE_TRANSFORM(x_hat)
            classify_df = classify_batch(x_hat_norm, y, classification_model, classify_df)


    # Average and save metrics
    print("RESULTS:")
    for k, v in metrics.items():
        metrics[k] = v / n_images
        print(f'{k}: {v / n_images}')
    if args.variable_rate:
        rd_path = osp.join(rd_dir, f'{args.eval_lambda}_{data_str}_{epoch}.json')
    else:
        rd_path = osp.join(rd_dir, f'{data_str}_{epoch}.json')
    with open(osp.join(rd_path), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)
        f.truncate()
    print(f'Saved results to {rd_path}.')

    if args.psd:
        # Average and save PSDs
        if args.psd_no_diff:
            psd_avg = psd_sum / n_images
            filename = osp.join(psd_dir, f'{data_str}.npy')
            np.save(filename, psd_avg)
            print(f'PSDs saved to {filename}')
        else:
            psd_diff_clean_avg = psd_diff_clean_sum / n_images
            if args.variable_rate:
                filename = osp.join(psd_dir, f'{args.eval_lambda}_{data_str}_vs_{clean_data_str}.npy')
            else:
                filename = osp.join(psd_dir, f'{data_str}_vs_{clean_data_str}.npy')
            np.save(filename, psd_diff_clean_avg)
            print(f'PSDs saved to {filename}')
            if args.corruption != 0:
                psd_diff_corrupt_avg = psd_diff_corrupt_sum / n_images
                if args.variable_rate:
                    filename = osp.join(psd_dir, f'{args.eval_lambda}_{data_str}_vs_{data_str}.npy')
                else:
                    filename = osp.join(psd_dir, f'{data_str}_vs_{data_str}.npy')
                np.save(filename, psd_diff_corrupt_avg)
                print(f'PSDs saved to {filename}')

    if args.classify:
        # Average and save
        accuracy = sum(classify_df['y_hat'] == classify_df['y']) / n_images
        if args.variable_rate:
            filename = osp.join(classify_dir, f'{args.eval_lambda}_{data_str}.csv')
        else:
            filename = osp.join(classify_dir, f'{data_str}.csv')
        f = open(filename, 'w')
        f.write(f"# Average accuracy: {accuracy:.2%}\n")
        classify_df.to_csv(f)
        f.close()
        print(f'Average accuracy: {accuracy:.2%}')