from .base_args import BaseArguments

class TrainArguments(BaseArguments):
    def initialize(self):
        
        # Data parameters
        self.parser.add_argument(
            '--train-folder', 
            default="CLIC/{}/train", 
            help='Folder to use for training data.'
            )
        self.parser.add_argument(
            '--val-folder', 
            default="CLIC/{}/valid", 
            help='Folder to use for validation data.'
            )

        # Rate parameters
        self.parser.add_argument(
            '--lmbda', 
            default=None,
            type=float, 
            help="Lambda value to use if training fixed-rate model."
            )
        self.parser.add_argument(
            '--lambda-range',
            default=(0.0012, 0.26),
            type=float,
            nargs=2,
            help="Range of lambdas to train over if training variable-rate model."
            )
        self.parser.add_argument(
            '--eval-lambdas',  
            nargs='+', 
            type=float, 
            help="Lambdas to use for validation on variable-rate models."\
                  "Length of these must equal batch_size.",
            default=(0.0012, 0.005, 0.01, 0.03, 0.05, 0.1, 0.15, 0.26,)
            )

        # Optimization parameters
        self.parser.add_argument(
            '--wt-lr', 
            default=1e-4, 
            type=float, 
            help="Learning rate for weight training optimizer."
            )
        self.parser.add_argument(
            '--aux-lr', 
            default=1e-3, 
            type=float, 
            help="Learning rate for auxillary optimizer."
            )
        self.parser.add_argument(
            "--no-grad-clip",
            default=False,
            action="store_true",
            help="Whether or not to clip gradients.",
            )

        # Batches and epochs
        self.parser.add_argument(
            "--batch-size",
            default=8,
            type=int,
            help="Training mini-batch size (default: 8)",
            )
        self.parser.add_argument(
            "--val-batch-size",
            default=2,
            type=int,
            help="Validation mini-batch size (default: 2). \
                This is smaller than the training because the full images are used for validation.",
            )
        self.parser.add_argument(
            "--patch-size",
            default=(256,256),
            type=int,
            nargs=2,
            help="Patch size to use for training. Will use random crops \
            of this size for training and center crops of this size for validation."
        )
        self.parser.add_argument(
            "--epochs",
            default=5000,
            type=int,
            help="number of total epochs to run",
            )


        # Pruning
        self.parser.add_argument(
            "--final-sparsity",
            default=0.95,
            type=float,
            help="Proportion of weights to prune.",
            )
        self.parser.add_argument(
            "--prune-epochs", 
            default=[675, 2400],
            nargs=2,
            type=int, 
            help="Initial and final epochs for pruning in GMP"
            )
        self.parser.add_argument(
            '--init-prune-sparsity',
            default=0.,
            type=float,
            help="Initial point to start gradually reducing sparsity from."
            )
        self.parser.add_argument(
            "--no-wt-bias", 
            action='store_true', 
            default=False,
            help="Whether to weight train the bias terms on pruned layers."
            )
        
        # Logging
        self.parser.add_argument(
            "--print-freq",
            default=10,
            type=int,
            help="How often to write to tensorbaord (in iterations)",
        )
        self.parser.add_argument(
            "--save-every", 
            default=100, 
            type=int, 
            help="How often to save checkpoint (in epochs)."
        )

        
        BaseArguments.initialize(self)