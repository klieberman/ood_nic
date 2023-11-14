import argparse
import yaml
import sys
import compressai

def trim_preceding_hyphens(st):
    i = 0
    while st[i] == "-":
        i += 1

    return st[i:]


def arg_to_varname(st: str):
    st = trim_preceding_hyphens(st)
    st = st.replace("-", "_")

    return st.split("=")[0]


def argv_to_vars(argv):
    var_names = []
    for arg in argv:
        if arg.startswith("-") and arg_to_varname(arg) != "config":
            var_names.append(arg_to_varname(arg))

    return var_names


def get_arg_str(args):
    s = '------------ Options -------------\n'
    for k, v in args.items():
        s += "{}: {}\n".format(k, v)
    s += '-------------- End ----------------\n'
    return s


class BaseArguments():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument(
            '--data-prefix',
            default='../../vision/datasets',
            help='Prefix for datasets (should include both clean and -C versions of datasets)'
        )
        self.parser.add_argument(
            "--run-dir", 
            default="runs/example",
            help="Path to save checkpoints, logs, configs, and results"
            )
        self.parser.add_argument(
            '--rep-count', 
            default=None, 
            help="Which run to save checkpoints and logs to. Helpful if resuming training over multiple jobs."
            )
        self.parser.add_argument(
            "--resume",
            default=None,
            type=str,
            help="path to latest checkpoint (default: none)",
            )
        
        # Compression arguments
        self.parser.add_argument(
            '--model',
            default='scale_hyperprior',
            choices=['scale_hyperprior', 'elic'],
            help="Model architecture to use."
        )
        self.parser.add_argument(
            '--N', 
            default=192, 
            type=int,  
            help="N parameter on scale hyperprior or ELIC model."
            )
        self.parser.add_argument(
            '--M', 
            default=192, 
            type=int,  
            help="M parameter on scale hyperprior or ELIC model."
            )
        self.parser.add_argument(
            '--variable-rate', 
            action='store_true', 
            default=False, 
            help="Whether to use a variable-rate version of the scale hyperprior model.")
        self.parser.add_argument(
            "--prune-algorithm", 
            default=None, 
            help="Algorithm to use for pruning: gmp (gradual magnitude pruning) or None"
            )
        self.parser.add_argument(
            '--no-prune-conv',
            default=False, 
            action='store_true', 
            help="Whether to prune convolutional layers."
            )
        self.parser.add_argument(
            "--no-prune-film",
            default=False, 
            action='store_true', 
            help="Whether to prune FiLM layers on variable-rate model."
            )
        self.parser.add_argument(
            "--layerwise", 
            action="store_true", 
            default=False, 
            help="Whether to prune network layerwise or globally. Default is globally."
            )
        self.parser.add_argument(
            '--distortion-metric',
            default="mse",
            choices=["mse", "ms-ssim"],
            help="Distortion metric to optimize for."
        )
        self.parser.add_argument(
            '--disable_tqdm',
            default=False,
            action='store_true',
            help="Whether to disable tqdm outputs."
        )
        
    
        # Hardware
        self.parser.add_argument(
            "--workers",
            default=2,
            type=int,
            help="number of data loading workers (default: 2)",
        )



    def parse(self):
        if not self.initialized:
            self.initialize()

        args, unknown = self.parser.parse_known_args()

        assert len(unknown) == 0, f'Unknown arguments {unknown}.'

        self.args = args
        self.arg_str = get_arg_str(vars(args))
        print(self.arg_str)

        return args


    def save(self, file_name):
        with open(file_name, 'wt') as opt_file:
            opt_file.write(self.arg_str)