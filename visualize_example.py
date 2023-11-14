from visualize.visualize import plot_psd, plot_fhm
from data.clic_info import IMAGE_SIZE


if __name__ == "__main__":
    plot_psd("runs/sh_fr_dense/lmbda-0.05/results_psds/clic_clean_vs_clic_clean.npy",
             IMAGE_SIZE,
             "example_psd.png"
             )
    # Requires results for Fourier heatmap shifts 0-2111
    plot_fhm("runs/sh_fr_dense/lmbda-0.05/results_rd/fhm/clic_fhm_{}_4999.json",
             "example_fhm.png")
    