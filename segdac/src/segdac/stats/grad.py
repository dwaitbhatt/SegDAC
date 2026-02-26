def compute_grad_stats(named_parameters):
    layers = []
    ave_grads = []
    max_grads = []
    l2_norms = []

    for name, p in named_parameters:
        if p.requires_grad and "bias" not in name:
            layers.append(name)
            if p.grad is not None:
                grad = p.grad.detach().cpu()
                ave_grads.append(grad.abs().mean().item())
                max_grads.append(grad.abs().max().item())
                l2_norms.append(grad.norm(2).item())
            else:
                ave_grads.append(0)
                max_grads.append(0)
                l2_norms.append(0)

    return layers, ave_grads, max_grads, l2_norms
