import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import os.path as osp
import time
import tqdm

from args.train_args import TrainArguments
from data.compression import ImageFolderCLIC
from models.scale_hyperprior import ScaleHyperprior
from models.scale_hyperprior_lct import ScaleHyperpriorLCT
from models.elic import ELICModel
from utils.compression import RateDistortionLoss
from utils.gmp import get_curr_sparsity_gmp, global_set_prune_thresholds, layerwise_set_sparsity_rates, check_sparsity
from utils.lct import get_lambdas, RateDistortionLossLCT
from utils.logging import AverageMeter, ProgressMeter, save_checkpoint
from utils.model_helpers import partition_params, resume
from utils.optimizers import get_optimizers, get_optimizer_states
from utils.utilities import get_device, makedirs_if_needed

'''
Script for training neural image compression models.
Based off of: https://github.com/InterDigitalInc/CompressAI/blob/14ac02c5182cbfee596abdfea98886be6247479a/examples/train.py
Has functionality to train fixed or variable rate models as well as dense or pruned (GMP) models and ELIC models.
'''

def train_one_epoch(train_loader, model, criterion, optimizers, epoch, curr_sparsity, args, writer, device):
    total_loss = AverageMeter("Loss", ":.3f")
    bpp_loss = AverageMeter("bpp_loss", ":6.4f")
    distortion_loss = AverageMeter("distortion_loss", ":6.4f")

    progress = ProgressMeter(
        len(train_loader),
        [total_loss, bpp_loss, distortion_loss],
        prefix=f"Epoch: [{epoch}]",
    )

    model.train()
    num_batches = len(train_loader)
    for i, images in tqdm.tqdm(enumerate(train_loader), ascii=True, total=num_batches, disable=args.disable_tqdm):
        images = images.to(device)

        # Set prune threshold or current sparsity rate
        if args.prune_algorithm == "gmp":
            if args.layerwise:
                layerwise_set_sparsity_rates(model, curr_sparsity, device)
            else:
                global_set_prune_thresholds(model, curr_sparsity, device)

            
        # Zero all gradients
        for _, optimizer in optimizers.items():
            optimizer.zero_grad()

        # Forward pass
        if args.variable_rate:
            lambdas = get_lambdas(*args.lambda_range, images.shape[0])
            lambdas = lambdas.to(device)
            out_net = model(images, lambdas)
            out_criterion = criterion(out_net, images, lambdas)
        else:
            out_net = model(images)
            out_criterion = criterion(out_net, images)

        # Backward pass on all optimizers
        out_criterion["loss"].backward()
        if not args.no_grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizers['wt'].step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        optimizers['aux'].step()
        
        # Record losses
        total_loss.update(out_criterion['loss'].item(), images.size(0))
        bpp_loss.update(out_criterion['bpp_loss'].item(), images.size(0))
        distortion_loss.update(out_criterion['distortion_loss'].item(), images.size(0))

        if i % args.print_freq == 0:
            # t = number of iterations
            t = num_batches * epoch + i
            progress.display(i)
            progress.write_to_tensorboard(writer, prefix="train", global_step=t)

    return total_loss.avg

def validate(val_loader, model, criterion, epoch,  args, writer, device):
    total_loss = AverageMeter("Loss", ":.3f")
    bpp_loss = AverageMeter("bpp_loss", ":6.4f")
    distortion_loss = AverageMeter("distortion_loss", ":6.4f")

    progress = ProgressMeter(
        len(val_loader),
        [total_loss, bpp_loss, distortion_loss],
        prefix=f"Epoch: [{epoch}]",
    )

    model.eval()
    num_batches = len(val_loader)
    if args.variable_rate:
        assert args.batch_size == len(args.eval_lambdas), "expected batch to have same size as evaluation lambdas for validation."
        val_lambdas = torch.tensor(args.eval_lambdas).unsqueeze(1)
        val_lambdas = val_lambdas.to(device)

    for i, images in tqdm.tqdm(enumerate(val_loader), ascii=True, total=num_batches):
        # Only validate on full batches because val_lambdas depends on this
        if len(images) == args.batch_size:
            images = images.to(device)

            # Forward pass
            if args.variable_rate:
                out_net = model(images, val_lambdas)
                out_criterion = criterion(out_net, images, val_lambdas)
            else:
                out_net = model(images)
                out_criterion = criterion(out_net, images)
            
            # Record losses
            total_loss.update(out_criterion['loss'].item(), images.size(0))
            bpp_loss.update(out_criterion['bpp_loss'].item(), images.size(0))
            distortion_loss.update(out_criterion['distortion_loss'].item(), images.size(0))

            if i % args.print_freq == 0:
                # t = number of iterations
                t = num_batches * epoch + i
                progress.display(i)
                progress.write_to_tensorboard(writer, prefix="val", global_step=t)

    return total_loss.avg



if __name__ == '__main__':
    train_args = TrainArguments()
    args = train_args.parse()

    device = get_device()

    if args.model == "scale_hyperprior":
        if args.variable_rate:
            print(f"Training variable-rate model over lambda range {args.lambda_range} \
            with {args.prune_algorithm} pruning algorithm.\n")
            
            model = ScaleHyperpriorLCT(N=args.N, M=args.M, args=args)
            
        else:
            assert args.lmbda is not None, "lmbda must be specified for fixed-rate models."
            print(f"Training fixed-rate model with lambda={args.lmbda} and pruning algorithm={args.prune_algorithm}.\n")
            
            model = ScaleHyperprior(N=args.N, M=args.M, args=args)
        
    elif args.model == "elic":
        model = ELICModel(N=args.N, M=args.M)
    else:
        exit(f'Error: invalid model choice {args.model}')

    model = model.to(device)
    if args.model == "scale_hyperprior" and args.variable_rate:
        criterion = RateDistortionLossLCT(metric=args.distortion_metric)
    else:
        criterion = RateDistortionLoss(args.lmbda, metric=args.distortion_metric)

    params = partition_params(model)
    optimizers = get_optimizers(args, model, params)
    
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizers['wt'], "min")

    if args.resume is not None:
        print(f"Resuming from {args.resume}.\n")
        model, optimizers, start_epoch, lr_scheduler = resume(args, model, optimizers, lr_scheduler)
    else:
        start_epoch = 0

    # Get relevant directories, save config file
    if args.rep_count is not None:
        run_dir = osp.join(args.run_dir, args.rep_count)
    else:
        run_dir = args.run_dir
    print(f"Saving checkpoints, configs, and logs in in {run_dir}.\n")
    log_dir = makedirs_if_needed(osp.join(run_dir, "logs"))
    checkpoint_dir = makedirs_if_needed(osp.join(run_dir, "checkpoints"))
    config_dir = makedirs_if_needed(osp.join(run_dir, "configs"))
    train_args.save(osp.join(config_dir, f"epoch_{start_epoch}.txt"))

    
    # Get dataset and loader (assumed to be CLIC)
    kwargs = {"num_workers": args.workers, "pin_memory": True}

    train_transform = transforms.Compose(
        [transforms.RandomCrop(args.patch_size, pad_if_needed=True), transforms.ToTensor()]
    )
    val_transform = transforms.Compose(
            [transforms.CenterCrop(args.patch_size), transforms.ToTensor()]
        )
    train_dataset = ImageFolderCLIC(osp.join(args.data_prefix, args.train_folder), transform=train_transform)
    val_dataset = ImageFolderCLIC(osp.join(args.data_prefix, args.val_folder), transform=val_transform)
    train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                **kwargs
            )
    val_loader = torch.utils.data.DataLoader(
                val_dataset, 
                batch_size=args.val_batch_size, 
                shuffle=False, 
                **kwargs
            )
    print(f"Train dataset has {len(train_dataset)} images ({len(train_loader)} batches of {args.batch_size}).")
    print(f"Validation dataset has {len(val_dataset)} images ({len(val_loader)} batches of {args.val_batch_size}).\n")


    # Train

    # Ensure 'gmp' has valid sparsity and prune epochs
    if args.prune_algorithm == "gmp":
        assert args.final_sparsity > 0 and args.final_sparsity < 1, 'invalid args.final_sparsity'
        assert args.prune_epochs[0] >= 0 and args.prune_epochs[0] < args.epochs, 'invalid starting prune epoch'
        assert args.prune_epochs[1] >= 0 and args.prune_epochs[1] < args.epochs, 'invalid ending prune epoch'
        assert args.prune_epochs[1] - args.prune_epochs[0] > 0, 'invalid range of prune epochs'

    writer = SummaryWriter(log_dir=log_dir)
    epoch_time = AverageMeter("epoch_time", ":.4f", write_avg=False)
    val_time = AverageMeter("validation_time", ":.4f", write_avg=False)
    train_time = AverageMeter("train_time", ":.4f", write_avg=False)
    progress_overall = ProgressMeter(
        1, [epoch_time, val_time, train_time], prefix="Overall Timing"
    )

    # Save the initial state
    save_checkpoint(
        epoch=0, 
        state_dict=model.state_dict(),
        optimizers=get_optimizer_states(optimizers),
        lr_scheduler=lr_scheduler.state_dict(),
        filename= osp.join(checkpoint_dir, "initial.state")
    )


    for epoch in range(start_epoch, args.epochs):
        start_epoch = time.time()

        # Get sparsity
        if args.prune_algorithm == 'gmp':
            curr_sparsity = get_curr_sparsity_gmp(epoch, args.prune_epochs, args.final_sparsity, args.init_prune_sparsity)
        elif args.prune_algorithm is None:
            curr_sparsity = 0.
        else:
            exit(f'args.prune_algorithm={args.prune_algorithm} is not implemented.')

        print(f'Epoch: {epoch}, current sparsity: {curr_sparsity}')

        # Train epoch
        start_train = time.time()
        train_one_epoch(train_loader, model, criterion, optimizers, epoch, curr_sparsity, args, writer, device)
        train_time.update((time.time() - start_train) / 60)

        if args.prune_algorithm == 'gmp':
            check_sparsity(model, curr_sparsity, layerwise=args.layerwise)

        # Validate epoch
        start_val = time.time()
        validate(val_loader, model, criterion, epoch, args, writer, device)
        val_time.update((time.time() - start_val) / 60)

        # Always save most recent state
        save_checkpoint(
            epoch=epoch, 
            state_dict=model.state_dict(),
            optimizers=get_optimizer_states(optimizers),
            lr_scheduler=lr_scheduler.state_dict(),
            filename= osp.join(checkpoint_dir, "most_recent.state")
        )

        # Save checkpoint every args.save_every epochs
        if ((epoch % args.save_every) == 0 and args.save_every > 0) or (epoch == args.epochs - 1):
            save_checkpoint(
                epoch=epoch,
                state_dict=model.state_dict(),
                optimizers=get_optimizer_states(optimizers),
                lr_scheduler=lr_scheduler.state_dict(),
                filename= osp.join(checkpoint_dir, f"epoch_{epoch:04d}.state"),
            )

        epoch_time.update((time.time() - start_epoch) / 60)
        progress_overall.display(epoch)
        progress_overall.write_to_tensorboard(
            writer, prefix="diagnostics", global_step=epoch
        )
    
    check_sparsity(model, args.final_sparsity, layerwise=args.layerwise)