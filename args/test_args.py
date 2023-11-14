from .base_args import BaseArguments
import compressai

class TestArguments(BaseArguments):
    
    def initialize(self):

        # Data information
        self.parser.add_argument(
            '--test-dataset', 
            default="kodak",
            choices=['kodak', 'clic', 'imagenet'],
            help="Determines which data parameters to use."
            )
        self.parser.add_argument(
            '--corruption',
            default=0,
            type=int,
            help="What corruption to use. \
            0 corresponds to clean, 1-15 correspond to corruptions in data_info."
        )
        self.parser.add_argument(
            '--severity',
            default="3",
            help="What severity to use."
        )
        self.parser.add_argument(
            "--batch-size",
            default=1,
            type=int,
            help="mini-batch size (default: 1)",
            )

        # Evaluation information
        self.parser.add_argument(
            '--no-compression',
            default=False,
            action='store_true',
            help="Whether to get results for images without compressing/reconstructing them."
        )
        self.parser.add_argument(
            '-c',
            '--classic-codec',
            default=None,
            choices=["jpeg2k", "vtm", "jpeg"],
            help="Whether to use a non-neural compression method."
        )
        self.parser.add_argument(
            '-q',
            '--quality',
            type=float,
            default=None,
            help="Quality for JPEG2000 or VTM compression."
        )
        self.parser.add_argument(
            '--vtm-build-dir',
            default= "../VVCSoftware_VTM-VTM-9.1/bin/umake/gcc-8.3/x86_64/release/"
        )
        self.parser.add_argument(
            '--vtm-config',
            default= "../VVCSoftware_VTM-VTM-9.1/cfg/encoder_intra_vtm.cfg"
        )
        self.parser.add_argument(
            '--eval-lambda',  
            type=float, 
            default=None,
            help="Lambda to test variable-rate model on. Must be specified if variable_rate is True.",
            )
        self.parser.add_argument(
            "--entropy-coder",
            choices=compressai.available_entropy_coders(),
            default=compressai.available_entropy_coders()[0],
            help="entropy coder (default: %(default)s)",
            )
        self.parser.add_argument(
            "--half",
            action="store_true",
            help="convert model to half floating point (fp16)",
            )
        self.parser.add_argument(
            "--entropy-estimation",
            action="store_true",
            help="use evaluated entropy estimation (no entropy coding)"
            )
        self.parser.add_argument(
            '--no-psnr-wrt-clean',
            default=False,
            action='store_true',
            help="Whether to calculate PSNR of reconstructed images with respect to their clean versions."
        )

        # PSD stuff
        self.parser.add_argument(
            '--psd',
            default=False,
            action='store_true',
            help="Whether to calculate PSDs."
        )
        self.parser.add_argument(
            '--psd-no-diff',
            default=False,
            action='store_true',
            help="Whether to calculate PSD without subtracting original images. \
            By default PSDs will be calculated by taking the difference between the reconstructed images and original images."
        )

        # Outputs to save
        self.parser.add_argument(
            '--save-images',
            default=False, 
            action='store_true',
            help="Whether to save reconstructed images."
            )
        self.parser.add_argument(
            '--classify',
            default=False,
            action='store_true',
            help="Whether to classify reconstructed images using pretrained ResNet."
        )

        # Fourier heatmap prameters
        self.parser.add_argument(
            '--fhm-idx',
            default=None, 
            type=int,
            help="Index of Fourier heatmap spectrum to use. \
            If None, no fourier augmentation will be done. \
            Can only be used with clean data (i.e., args.corruption = 0)"
            )
        self.parser.add_argument(
            '--fhm-size',
            default=[64,32], 
            type=int,
            nargs=2,
            help="Size of Fourier heatmap to generate."
            )
        self.parser.add_argument(
            '--subset',
            default=False,
            action='store_true',
            help="Whether to use a subset of images (only valid for ImageNet)."
        )


        BaseArguments.initialize(self)