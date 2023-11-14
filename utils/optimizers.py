import torch

def get_optimizers(args, model, params, debug=False):
    if debug:
        for n, v in model.named_parameters():
            if v.requires_grad:
                print("<DEBUG> gradient to", n)

            if not v.requires_grad:
                print("<DEBUG> no gradient to", n)

    params_full_dict = dict(model.named_parameters())
    optimizers = {}

    # Always have a separate auxillary optimizer (Adam)
    optimizers['aux'] = torch.optim.Adam(
        (params_full_dict[n] for n in sorted(params['aux'])),
        lr=args.aux_lr,
    )
    print(f"Adam auxillary optimizer with lr={args.aux_lr}.")

    # Always have a weight training optimizer
    optimizers['wt'] = torch.optim.Adam(
            (params_full_dict[n] for n in sorted(params['wt'])),
            lr=args.wt_lr,
        )
    print(f"Adam weight training optimizer with lr={args.wt_lr}.\n")

    return optimizers

    

def get_optimizer_states(optimizers):
    optimizer_states = {}
    for k, v in optimizers.items():
        optimizer_states[k] = v.state_dict()
    return optimizer_states