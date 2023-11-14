# Neural Image Compression: Generalization, Robustness, and Spectral Biases

Official code implementation for ["Neural Image Compression: Generalization, Robustness, and Spectral Biases".](https://arxiv.org/abs/2307.08657)

## Environment
Details about the environment are specified in `environment.yml`.

## Data (CLIC, Kodak, and -C variants)
The 2020 Challenge on Learned Image Compression (CLIC) dataset can be downloaded from these links:

https://data.vision.ee.ethz.ch/cvl/clic/mobile_train_2020.zip
https://data.vision.ee.ethz.ch/cvl/clic/professional_train_2020.zip
https://data.vision.ee.ethz.ch/cvl/clic/mobile_valid_2020.zip
https://data.vision.ee.ethz.ch/cvl/clic/professional_valid_2020.zip
https://data.vision.ee.ethz.ch/cvl/clic/test/CLIC2020Mobile_test.zip
https://data.vision.ee.ethz.ch/cvl/clic/test/CLIC2020Professional_test.zip

The Kodak dataset can be found here: https://r0k.us/graphics/kodak/.

The -C datasets were generated using code from: https://github.com/hendrycks/robustness/tree/master.

## Training NIC models
`train.py` contains code to train the following neural image compression (NIC) models. 
- [Scale-hyperprior](https://arxiv.org/abs/1802.01436)
- [ELIC](https://arxiv.org/abs/2203.10886)
- A variable-rate version of scale-hyperprior trained using [loss conditional training](https://openreview.net/pdf?id=HyxY6JHKwr).
- A sparse version of scale-hyperprior pruned using [gradual magnitude pruning](https://arxiv.org/abs/1710.01878). 

By default models are trained to optimize PSNR, but they can be trained to optimize MS-SSIM using the `--distortion-metric` flag. The training datasets and loaders are compatible with the CLIC dataset. See arguments in `args/base_args.py` and `args/train_args.py` for more details.

### Example commands
Train a fixed-rate scale-hyperprior model:
```
python train.py --lmbda 0.05 --run-dir runs/sh_fr_dense/lmbda-0.05 --epochs 5000
```

Train an ELIC model:
```
python train.py --model elic  --N 192 --M 320 --lmbda 0.05 --run-dir runs/elic/lmbda-0.05 --epochs 5000 
```

Train a variable-rate scale-hyperprior model:
```
python train.py --run-dir runs/sh_vr_dense --variable-rate --lambda-range 0.001 0.25 --epochs 10000
```

Train a 95% sparse scale-hyperprior model:
```
python train.py --lmbda 0.05 --run-dir runs/sh_fr_pruned/sparsity-0.95/lmbda-0.05 --prune-algorithm gmp --final-sparsity 0.95
```


## Testing models
`test.py` contains code to test image compression codecs.
- Image compressor options include NIC models listed above or classic codecs (i.e., JPEG, JPEG2000, and VTM (equivalently VVC)). 
    - If testing a NIC model, use `--resume` to specify a checkpoint for testing. 
        - If testing a variable-rate NIC model, use `--variable-rate` and `--eval-lambda` to test the model on a particular lambda.
    - If testing a classic codec, use `--classic-codec` and `-q` to specify which codec and quality. 
    - If testing without compression, use `--no-compression`.
- Dataset choices are CLIC, Kodak, or ImageNet (validation split) and their -C versions. 
    - Use `--test-dataset` to specify which dataset to use. 
    - To test the clean dataset (e.g., CLIC), set `--corruption` and `--severity` to 0.
    - To test a corruption from the -C dataset, set `--corruption` to the number 1-15 corresponding to the desired corruption (the order of corruptions can be found in `data/corruptions.py`) and `--severity` to the number 1-5 corresponding to the desired severity.
    - To test a Fourier-shifted dataset, use `--fhm-idx`.
- By default, `test.py` will save the rate-distortion (bpp/PSNR) metrics in a json file. Additional arguments for more evaluation functionality include:
    - `--psd`: save the arrays of the PSDs for the model on that dataset. If the corruption is clean (i.e., corruption=0), then one PSD ($\mathcal{D}$) will be saved. If the corruption is not clean (i.e., using OOD data), then two PSDs ($\mathcal{G}$ and $\mathcal{R}$) will be saved.
    - `--save-images`: saves reconstructed images.
    - ``--classify``: use a pre-trained ResNet model to classify the images after compression. Can only be used with ImageNet dataset.
- See arguments in `args/test_args.py` for more details. 

### Example commands

Test pre-trained SH NIC model on CLIC dataset: 
```
python test.py --run-dir runs/sh_fr_dense/lmbda-0.05 --resume runs/sh_fr_dense/lmbda-0.05/checkpoints/epoch_4999.state --test-data clic --corruption 0 --severity 0 --psd
```

Test JPEG2000 on shot noise corruption of Kodak-C dataset:
```
python test.py --run-dir runs/jpeg2k/quality-20 --classic-codec jpeg2k -q 20 --test-data kodak --corruption 14 --severity 3 --psd
```

To get the Fourier heatmaps, we need to test model on all Fourier-shifted versions datasets. We use Fourier heatmaps of size 64 x 64. Because of their symmetry, we need to test the model on 33 * 64 = 2112 of these shifts. To test a particular shift, use the `--fhm-idx` flag as shown below:
```
for i in {0..2111}
do
    python test.py --run-dir runs/sh_fr_dense/lmbda-0.05 --resume runs/sh_fr_dense/lmbda-0.05/checkpoints/epoch_4999.state --fhm-idx $i
done
```

## Visualizing D, G, R and Fourier Heatmaps
Code for visualizing PSDS ($\mathcal{D}, \mathcal{G}, \mathcal{R}$) and Fourier Heatmaps is in `visualize/visualize.py`. An example of how to use these is in `visualize_example.py`.


## Citation
If you use our work, please cite:
```
@article{lieberman2023neural,
      title={Neural Image Compression: Generalization, Robustness, and Spectral Biases}, 
      author={Lieberman, Kelsey and Diffenderfer, James and Godfrey, Charles and Kailkhura, Bhavya},
      journal={NeurIPS},
      year={2023},
}
```

## License
This repository is distributed under the terms of the MIT license. LLNL-CODE-856963.

