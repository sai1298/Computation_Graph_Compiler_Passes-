# Project 3: Computation Graph Optimization Pass (Operator Fusion + Constant Folding)

## Goal
Implement real compiler-style optimization passes (operator fusion, constant folding) on a PyTorch model's computation graph using `torch.fx`, and measure actual execution time improvement. Targets Nvidia/Qualcomm JD language around "ML compilers," "graph optimization," and "kernel code generators."

## Tech stack
- Python 3.10+
- PyTorch + `torch.fx` (graph capture and transformation)
- `torch.jit` or `torch.compile` for comparison baseline (optional, to contextualize your hand-written passes against PyTorch's own compiler)
- matplotlib for result visualization

## Conceptual background the agent should apply
- `torch.fx` traces a model into a graph of nodes (`call_module`, `call_function`, `call_method`, `placeholder`, `output`).
- **Constant folding**: any subgraph whose inputs are all constants (e.g., `x = 2; y = 3; z = x + y`) can be pre-computed at "compile time" and replaced with a single constant node.
- **Operator fusion**: consecutive elementwise ops (e.g., `Conv2d -> BatchNorm -> ReLU`, or `Linear -> ReLU`) can be fused into a single fused module/op to reduce kernel-launch overhead and intermediate memory traffic.

## Step-by-step implementation plan
1. **Setup**: Install torch (with fx included by default).
2. **Choose target model**: A small CNN (few Conv2d+BatchNorm+ReLU blocks) is ideal since Conv-BN-ReLU fusion is a well-known, well-defined pattern.
3. **Trace the model**: Use `torch.fx.symbolic_trace(model)` to get a `GraphModule`.
4. **Implement Conv-BN fusion pass**: Detect `Conv2d` immediately followed by `BatchNorm2d` in the graph, mathematically fold the BatchNorm parameters into the Conv2d's weight and bias (this is a standard, well-documented technique — implement the actual math, don't just call a built-in helper without understanding it), and rewire the graph to remove the BatchNorm node.
5. **Implement constant folding pass**: Walk the graph, identify nodes whose all inputs are literals/constants (no dependency on the `placeholder` input), evaluate them eagerly, and replace with a constant node.
6. **Recompile**: Call `graph_module.recompile()` after each transformation so the generated Python matches the modified graph.
7. **Correctness check**: Verify the optimized model produces numerically identical (within floating point tolerance, e.g., `torch.allclose`) outputs to the original model on a batch of random inputs. This is non-negotiable — an optimization that changes outputs is a bug, not an optimization.
8. **Benchmark**: Measure inference latency (many runs, report p50/p95) before and after each optimization pass, on CPU.
9. **README**: Explain each pass with before/after graph printouts (`graph_module.graph.print_tabular()`), correctness verification results, and real latency numbers.

## Deliverables
- `model_def.py` — small CNN definition (few conv/bn/relu blocks)
- `passes/fuse_conv_bn.py` — Conv-BN fusion pass implementation
- `passes/constant_fold.py` — constant folding pass implementation
- `benchmark.py` — correctness check + latency benchmarking before/after
- `README.md` — explanation of each pass, graph before/after printouts, real benchmark numbers
- `results/` — raw timing logs

## Metrics

- **Inference Latency:** Report p50 and p95 latency in milliseconds over 100+ runs for:
  - Original model
  - After Conv-BN fusion
  - After constant folding
  - After both optimizations combined

- **Graph Size:** Number of `torch.fx` graph nodes before and after each optimization pass.

- **Optimization Impact:** Number of Conv-BN patterns fused and number of constant expressions folded.

- **Correctness:** Maximum absolute difference between original and optimized model outputs, along with the `torch.allclose()` result.

- **Speedup:** Percentage latency improvement and/or speedup calculated as:

  `Speedup = Original Latency / Optimized Latency`

- **Hardware:** CPU model, number of CPU cores/threads, RAM, operating system, Python version, and PyTorch version used for benchmarking.

- **Benchmark Configuration:** Input tensor shape, batch size, warm-up iterations, measured iterations, and number of PyTorch CPU threads used.
