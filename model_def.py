"""Target model for the FX optimization passes.

Deliberately contains BOTH optimization opportunities:

1. Four Conv2d -> BatchNorm2d -> ReLU blocks. Conv-BN is the canonical fusion
   pattern: in eval mode BatchNorm is a fixed affine map, so it can be folded
   into the preceding convolution's weights exactly.

2. A short chain of arithmetic on constant buffers (`c1 * c2 + c3`). Under FX
   these become `get_attr` nodes feeding `call_function` nodes with no
   dependency on the input placeholder -- i.e. computable at "compile" time.

The model is kept small (4 blocks) so the graph is easy to read in
`print_tabular()` output, which is itself evidence the passes did something.
"""

import torch
import torch.nn as nn


class ConvBNReLU(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class SmallCNN(nn.Module):
    """4x (Conv-BN-ReLU) + a constant-foldable scaling tail + classifier head."""

    def __init__(self, num_classes: int = 10, width: int = 32):
        super().__init__()
        self.block1 = ConvBNReLU(3, width)
        self.block2 = ConvBNReLU(width, width * 2)
        self.pool1 = nn.MaxPool2d(2)
        self.block3 = ConvBNReLU(width * 2, width * 2)
        self.block4 = ConvBNReLU(width * 2, width * 4)
        self.pool2 = nn.MaxPool2d(2)

        # Constants stored as buffers so FX sees them as `get_attr` nodes rather
        # than folding them away at trace time (a Python literal like 2*3 would
        # already be constant-folded by the interpreter before FX ever sees it,
        # which would make the pass look like it worked when it did nothing).
        self.register_buffer("c1", torch.tensor(2.0))
        self.register_buffer("c2", torch.tensor(3.0))
        self.register_buffer("c3", torch.tensor(0.5))

        self.head = nn.Linear(width * 4, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool1(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool2(x)

        # --- constant subgraph: no dependency on `x`, so it is foldable ---
        k = self.c1 * self.c2      # -> 6.0
        k = k + self.c3            # -> 6.5
        k = k / self.c1            # -> 3.25
        # --- end constant subgraph ---

        x = x * k                  # depends on input: NOT foldable
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.head(x)


def build_model(seed: int = 42) -> nn.Module:
    """Eval-mode model with deterministic weights.

    BN fusion is only mathematically valid in eval mode, where BatchNorm uses
    its stored running statistics instead of batch statistics. The model is
    also given a few forward passes of random data first so the running stats
    are not the identity defaults -- otherwise the fusion would be trivially
    correct and would not actually test the folding math.
    """
    torch.manual_seed(seed)
    model = SmallCNN()
    model.train()
    with torch.no_grad():
        for _ in range(10):
            model(torch.randn(8, 3, 32, 32))
    model.eval()
    return model
