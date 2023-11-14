import torch
import torch.nn.functional as F

from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import GDN
from compressai.models.google import CompressionModel, get_scale_table
from compressai.models.utils import update_registered_buffers

from utils.builder import get_builder  


class ScaleHyperprior(CompressionModel):
    """Scale Hyperprior model from J. Balle, D. Minnen, S. Singh, S.J. Hwang,
    N. Johnston: `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_ Int. Conf. on Learning Representations
    (ICLR), 2018.

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
        bias (bool): Whether to include bias terms on the convolutional layers
    """

    def __init__(self, N=192, M=192, bias=True, args=None):
        super().__init__()
        self.entropy_bottleneck = EntropyBottleneck(N)
        print(f'Building ScaleHyperprior with N={N}, M={M}')
        builder = get_builder(args)
        
         # Analysis transform
        self.conv_ga1 = builder.conv(5, 3, N, stride=2, bias=bias)
        self.gdn_ga1 = GDN(N)
        self.conv_ga2 = builder.conv(5, N, N, stride=2, bias=bias)
        self.gdn_ga2 = GDN(N)
        self.conv_ga3 = builder.conv(5, N, N, stride=2, bias=bias)
        self.gdn_ga3 = GDN(N)
        self.conv_ga4 = builder.conv(5, N, M, stride=2, bias=bias)

        # Hyper-analyis transform
        self.conv_ha1 = builder.conv(3, M, N, stride=1, bias=bias)
        self.conv_ha2 = builder.conv(5, N, N, stride=2, bias=bias)
        self.conv_ha3 = builder.conv(5, N, N, stride=2, bias=False)

        # Hyper-synthesis transform
        self.deconv_hs1 = builder.deconv(5, N, N, stride=2, bias=bias)
        self.deconv_hs2 = builder.deconv(5, N, N, stride=2, bias=bias)
        self.deconv_hs3 = builder.deconv(3, N, M, stride=1, bias=bias)

        # Synthesis transform
        self.deconv_gs1 = builder.deconv(5, M, N, stride=2, bias=bias)
        self.igdn_gs1 = GDN(N, inverse=True)
        self.deconv_gs2 = builder.deconv(5, N, N, stride=2, bias=bias)
        self.igdn_gs2 = GDN(N, inverse=True)
        self.deconv_gs3 = builder.deconv(5, N, N, stride=2, bias=bias)
        self.igdn_gs3 = GDN(N, inverse=True)
        self.deconv_gs4 = builder.deconv(5, N, 3, stride=2, bias=bias)
        
        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)
        

    def ga(self, x):
        x = self.conv_ga1(x)
        x = self.gdn_ga1(x)
        x = self.conv_ga2(x)
        x = self.gdn_ga2(x)
        x = self.conv_ga3(x)
        x = self.gdn_ga3(x)
        x = self.conv_ga4(x)
        return x

    def gs(self, x):
        x = self.deconv_gs1(x)
        x = self.igdn_gs1(x)
        x = self.deconv_gs2(x)
        x = self.igdn_gs2(x)
        x = self.deconv_gs3(x)
        x = self.igdn_gs3(x)
        x = self.deconv_gs4(x)
        return x

    def ha(self, x):
        x = self.conv_ha1(x)
        x = F.relu(x)
        x = self.conv_ha2(x)
        x = F.relu(x)
        x = self.conv_ha3(x)
        return x

    def hs(self, x):
        x = self.deconv_hs1(x)
        x = F.relu(x)
        x = self.deconv_hs2(x)
        x = F.relu(x)
        x = self.deconv_hs3(x)
        return x



    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)

    def forward(self, x):
        y = self.ga(x)
        z = self.ha(torch.abs(y))
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        scales_hat = self.hs(z_hat)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat)
        x_hat = self.gs(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }
    

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["conv_ga1.weight"].size(0)
        M = state_dict["conv_ga4.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated

    def compress(self, x):
        y = self.ga(x)
        z = self.ha(torch.abs(y))

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        scales_hat = self.hs(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        scales_hat = self.hs(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, z_hat.dtype)
        x_hat = self.gs(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}