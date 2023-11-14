import torch
from collections import OrderedDict

import os.path as osp
from compressai.zoo import load_state_dict

from utils.gmp import get_curr_sparsity_gmp, check_sparsity

def partition_params(model):

    params_dict = dict((n, p) for n, p in model.named_parameters() \
        if (not n.endswith('.prune_threshold') and not n.endswith('.curr_sparsity_rate')))
    
    # Quantiles are always auxillary parameters
    aux_parameters= {
            n
            for n, p in model.named_parameters()
            if n.endswith(".quantiles") and p.requires_grad
    }
        
    # Entropy and GDN parameters are always weight trained
    entropy_parameters = {
        n
        for n, p in model.named_parameters()
        if n.startswith("entropy") and not n.endswith(".quantiles") and p.requires_grad
    }
    gdn_parameters = {
        n
        for n, p in model.named_parameters()
        if (n.startswith('gdn') or n.startswith('igdn')) and p.requires_grad
    }
        
    # FiLM and conv parameters are always weight trained in GMP (not true with biprop/edge popup)
    film_parameters = {
                n
                for n, p in model.named_parameters()
                if n.startswith("film") and p.requires_grad
    }
    conv_parameters = {
                n
                for n, p in model.named_parameters()
                if (n.startswith("conv") or n.startswith("deconv") \
                    or n.startswith('g_a') or n.startswith('g_s') \
                    or n.startswith('h_a') or n.startswith('h_s')) and p.requires_grad
    }
    charm_parameters = {
                n
                for n, p in model.named_parameters()
                if (n.startswith("charm")) and p.requires_grad
    }
    elic_parameters = {
        n
        for n, p in model.named_parameters()
        if (n.startswith('cc_transforms') or n.startswith('context_prediction') \
            or n.startswith('ParamAggregation'))
    }

    param_counts = {}
    n_total = 0
    for name, param_set in zip(['conv', 'film', 'entropy', 'gdn', 'aux', 'charm', 'elic'], \
        [conv_parameters, film_parameters, entropy_parameters, gdn_parameters, aux_parameters, charm_parameters, elic_parameters]):
        n_set = sum(int(params_dict[n].numel()) for n in param_set if not n.endswith('scores'))
        param_counts[name] = n_set
        n_total += n_set
    print(f'Total number of parameters: {n_total}.')
    for k, v in param_counts.items():
        print(f'{v} {k} parameters ({v/n_total:.2%})')
    print('')

    wt_parameters = entropy_parameters | gdn_parameters | conv_parameters | film_parameters | charm_parameters | elic_parameters
    
    # Make sure we don't have an intersection of parameters
    inter_params = aux_parameters & wt_parameters
    union_params = aux_parameters | wt_parameters
    assert len(inter_params) == 0
    for param in params_dict.keys():
        if param not in union_params:
            print(param)
    
    assert len(union_params) - len(params_dict.keys()) == 0
   
    return {'aux': aux_parameters, 'wt': wt_parameters}


def resume(args, model, optimizers, lr_scheduler, confirm_sparsity=True):
    assert osp.isfile(args.resume), f"{args.resume} is not a file"

    checkpoint = torch.load(args.resume, map_location=f"cuda:0")
    
    # If checkpoint only contains state_dict
    if 'entropy_bottleneck._matrix0' in checkpoint.keys():
        state_dict = load_state_dict(checkpoint)
        model.load_state_dict(state_dict)
        optimizers = None
        epoch = None
    # If keys of state_dict start with module. (i.e., model was saved with DataParllel)
    elif "module.patch_embed.proj.weight" in checkpoint['state_dict'].keys():
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        optimizers = None
        epoch = None
    else:
        model.load_state_dict(checkpoint["state_dict"])

    if optimizers is not None:
        for k in optimizers.keys():
            optimizers[k].load_state_dict(checkpoint["optimizers"][k])
    if 'epoch' in checkpoint.keys():
        epoch = checkpoint['epoch']
    else:
        epoch = None

    if lr_scheduler is not None:
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    print(f"=> Loaded checkpoint '{args.resume}' (epoch {epoch})")

    if confirm_sparsity:
        if args.prune_algorithm == 'gmp':
                curr_sparsity = get_curr_sparsity_gmp(epoch, args.prune_epochs, args.final_sparsity, args.init_prune_sparsity)
        elif args.prune_algorithm is None:
            curr_sparsity = 0.
        else:
            exit(f'args.prune_algorithm={args.prune_algorithm} is not implemented.')

        if curr_sparsity > 0.:
            check_sparsity(model, curr_sparsity, layerwise=args.layerwise)

    return model, optimizers, epoch, lr_scheduler