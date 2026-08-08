"""Constant-folding pass over a torch.fx graph.

Algorithm
---------
1. Mark every `placeholder` node (the model's runtime inputs) as NON-constant.
2. Propagate: a node is constant iff all of its inputs are constant.
   `get_attr` nodes -- references to parameters and buffers -- are the leaves
   that seed the constant set.
3. Evaluate each maximal constant subgraph once, eagerly, and register the
   result as a new buffer.
4. Replace uses of the subgraph's output with a `get_attr` on that buffer, then
   dead-code-eliminate the now-unused nodes.

The "maximal" part matters. Folding every constant node individually would
replace `c1*c2` with a buffer and `+c3` with another buffer and gain nothing.
Only the *last* node of a constant chain -- one whose users include at least one
non-constant node -- is materialised.

Deliberately conservative: `call_module` nodes are never folded even when their
inputs are constant, since a module could carry state (dropout, running stats)
that makes eager evaluation unsound.
"""

from __future__ import annotations

import copy

import torch
import torch.fx as fx


def _is_foldable_op(node: fx.Node) -> bool:
    # get_attr seeds the constant set; call_function / call_method propagate it.
    return node.op in ("get_attr", "call_function", "call_method")


def constant_fold(gm: fx.GraphModule) -> tuple[fx.GraphModule, dict]:
    """Fold constant subgraphs. Returns (module, stats)."""
    gm = copy.deepcopy(gm)

    # ---- step 1 + 2: mark which nodes are constant -------------------------
    is_const: dict[fx.Node, bool] = {}
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            is_const[node] = False
        elif node.op == "output":
            is_const[node] = False
        elif node.op == "get_attr":
            is_const[node] = True
        elif _is_foldable_op(node):
            inputs = list(node.all_input_nodes)
            is_const[node] = bool(inputs) and all(is_const.get(i, False) for i in inputs)
        else:
            is_const[node] = False  # call_module: conservatively not folded

    const_nodes = [n for n, c in is_const.items() if c and n.op != "get_attr"]

    # ---- step 3: find the frontier -- constant nodes with a non-constant user
    frontier = [
        n for n in const_nodes
        if any(not is_const.get(u, False) for u in n.users)
    ]

    if not frontier:
        return gm, {"folded_subgraphs": 0, "nodes_removed": 0, "values": {}}

    # Evaluate the whole graph's constant part once via an FX interpreter run.
    # Any real input works -- constant nodes ignore it by construction.
    env: dict[fx.Node, object] = {}

    def _resolve(a):
        return fx.node.map_arg(a, lambda n: env[n])

    for node in gm.graph.nodes:
        if not is_const.get(node, False):
            continue
        if node.op == "get_attr":
            env[node] = _fetch_attr(gm, node.target)
        elif node.op == "call_function":
            env[node] = node.target(*_resolve(node.args), **_resolve(node.kwargs))
        elif node.op == "call_method":
            self_obj, *rest = _resolve(node.args)
            env[node] = getattr(self_obj, node.target)(*rest, **_resolve(node.kwargs))

    # ---- step 4: materialise and rewire ------------------------------------
    values = {}
    for i, node in enumerate(frontier):
        value = env[node]
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value)
        buf_name = f"_folded_const_{i}"
        gm.register_buffer(buf_name, value.detach().clone())
        values[buf_name] = value.detach().clone().flatten().tolist()

        with gm.graph.inserting_before(node):
            new_node = gm.graph.get_attr(buf_name)
        node.replace_all_uses_with(new_node)

    before = len(list(gm.graph.nodes))
    gm.graph.eliminate_dead_code()
    gm.graph.lint()
    gm.recompile()
    after = len(list(gm.graph.nodes))

    return gm, {
        "folded_subgraphs": len(frontier),
        "nodes_removed": before - after,
        "values": values,
    }


def _fetch_attr(mod: torch.nn.Module, target: str):
    obj = mod
    for part in target.split("."):
        obj = getattr(obj, part)
    return obj
