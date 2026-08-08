"""Conv2d + BatchNorm2d fusion pass over a torch.fx graph.

Scope, stated honestly: FX supplies the graph representation and the tracer.
The pattern matching, the folding mathematics, the construction of the
replacement module, and the graph rewiring below are hand-implemented. This is
"I wrote the optimization pass", not "I wrote a compiler".

The mathematics
---------------
In eval mode BatchNorm2d is a fixed per-channel affine transform:

    bn(y) = gamma * (y - mean) / sqrt(var + eps) + beta

and the convolution it follows is:

    y = W * x + b

Substituting, with  s = gamma / sqrt(var + eps)  (a per-output-channel scalar):

    bn(conv(x)) = s * (W * x + b - mean) + beta
                = (s * W) * x + (s * (b - mean) + beta)

So the fused convolution uses:

    W' = W * s        broadcast over the output-channel dimension
    b' = (b - mean) * s + beta

This is exact in eval mode -- not an approximation -- which is why the
correctness check downstream expects agreement to floating-point tolerance
rather than merely "close".
"""

from __future__ import annotations

import copy

import torch
import torch.fx as fx
import torch.nn as nn


def fuse_conv_bn_weights(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """Return a new Conv2d equivalent to `bn(conv(x))`, with BN folded in."""
    assert not bn.training, "BN fusion is only valid in eval mode"

    fused = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=True,  # fusion always produces a bias, even if the conv had none
        padding_mode=conv.padding_mode,
    )

    w = conv.weight.detach().clone()
    b = (
        conv.bias.detach().clone()
        if conv.bias is not None
        else torch.zeros(conv.out_channels, dtype=w.dtype, device=w.device)
    )

    gamma = bn.weight.detach().clone()
    beta = bn.bias.detach().clone()
    mean = bn.running_mean.detach().clone()
    var = bn.running_var.detach().clone()
    eps = bn.eps

    s = gamma / torch.sqrt(var + eps)              # per-output-channel scale

    # w has shape [out, in/groups, kh, kw]; s must broadcast over dim 0 only.
    fused.weight.data = w * s.reshape(-1, 1, 1, 1)
    fused.bias.data = (b - mean) * s + beta
    return fused


def _module_of(model: nn.Module, target: str) -> nn.Module:
    mod = model
    for part in target.split("."):
        mod = getattr(mod, part)
    return mod


def _set_module(model: nn.Module, target: str, new: nn.Module) -> None:
    parts = target.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new)


def fuse_conv_bn(gm: fx.GraphModule) -> tuple[fx.GraphModule, int]:
    """Rewrite every Conv2d->BatchNorm2d pair in `gm`. Returns (module, n_fused).

    A pair is only fused when the BatchNorm consumes the convolution's output
    and *nothing else does*. If the conv output has another consumer (a skip
    connection, say), folding would silently change that other branch's input,
    so those pairs are left alone.
    """
    gm = copy.deepcopy(gm)
    modules = dict(gm.named_modules())
    n_fused = 0

    for node in list(gm.graph.nodes):
        if node.op != "call_module":
            continue
        if not isinstance(modules.get(node.target), nn.BatchNorm2d):
            continue

        (conv_node,) = node.args if len(node.args) == 1 else (node.args[0],)
        if conv_node.op != "call_module":
            continue
        conv = modules.get(conv_node.target)
        if not isinstance(conv, nn.Conv2d):
            continue

        # Guard: the conv feeds this BN and nothing else.
        if len(conv_node.users) != 1:
            continue

        bn = modules[node.target]
        fused = fuse_conv_bn_weights(conv, bn)

        _set_module(gm, conv_node.target, fused)
        modules[conv_node.target] = fused

        # Rewire: everything that read the BN output now reads the conv output.
        node.replace_all_uses_with(conv_node)
        gm.graph.erase_node(node)
        n_fused += 1

    gm.graph.lint()
    gm.recompile()  # regenerate forward() so the emitted Python matches the graph
    return gm, n_fused
