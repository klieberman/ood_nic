# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch
import math
from scipy.stats import loguniform
from pytorch_msssim import ms_ssim

from utils.compression import RateDistortionLoss


def get_lambdas(a, b, n):
    lambdas = loguniform.rvs(a, b, size=n)
    lambdas = torch.Tensor(lambdas).unsqueeze(1)
    return lambdas


class RateDistortionLossLCT(RateDistortionLoss):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, metric="mse", return_type="all"):
        super().__init__(lmbda=None, metric=metric, return_type=return_type)

    def forward(self, output, target, lambdas):    
        N, _, H, W = target.size()
        assert lambdas.shape[0] == N, "size of lmbda does not match size of x."

        loss = 0
        for i in range(N):
            lambda_i = lambdas[i]
            x_hat_i = output["x_hat"][i]
            target_i = target[i]
            num_pixels_i = H * W
            likelihoods_i = [output["likelihoods"]["y"][i], output["likelihoods"]["z"][i]]
            bpp_loss_i = sum(
                (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels_i))
                for likelihoods in likelihoods_i
            )
            if self.metric == ms_ssim:
                x_hat_i = torch.unsqueeze(x_hat_i, 0)
                target_i = torch.unsqueeze(target_i, 0)               
                distortion_i = 4000 * (1 - self.metric(x_hat_i, target_i, data_range=1))
            else:
                distortion_i = 255**2 * self.metric(x_hat_i, target_i)

            # print(f"lmbda={lambda_i}, distortion={distortion_i}, bpp={bpp_loss_i}")
            loss += lambda_i * distortion_i + bpp_loss_i

        out = {}
        out["loss"] = loss
        num_pixels = N * H * W
        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        if self.metric == ms_ssim:
            out["distortion_loss"] = self.metric(output["x_hat"], target, data_range=1)
        else:
            out["distortion_loss"] = self.metric(output["x_hat"], target)

        if self.return_type == "all":
            return out
        else:
            return out[self.return_type]



