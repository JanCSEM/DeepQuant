# ...existing code...
import onnx
import onnx_graphsurgeon as gs
import numpy as np


_LINEAR_OPS = {"Conv", "Gemm", "MatMul"}
_POOL_OPS = {"MaxPool", "AveragePool", "GlobalAveragePool"}
_NONLINEAR_OPS = {"Gelu", "Softmax", "Sigmoid", "Tanh"}
_NORM_OPS = {"BatchNorm", "LayerNorm", "GroupNorm"}

def _is_const(x):
    return isinstance(x, gs.Constant)


def _non_const_input(node: gs.Node):
    for i in node.inputs:
        if isinstance(i, gs.Variable):
            return i
    return None


def _const_input(node: gs.Node):
    for i in node.inputs:
        if _is_const(i):
            return i
    return None


def _is_mul_dequant(node: gs.Node) -> bool:
    return node is not None and node.op == "Mul" and len(node.inputs) == 2 and (_is_const(node.inputs[0]) ^ _is_const(node.inputs[1]))


def _build_maps(graph: gs.Graph):
    producers, consumers = {}, {}
    for n in graph.nodes:
        for o in n.outputs:
            producers[o.name] = n
        for i in n.inputs:
            if isinstance(i, gs.Variable):
                consumers.setdefault(i.name, []).append(n)
    return producers, consumers


def _replace_all_consumers(graph: gs.Graph, old_var: gs.Variable, new_var: gs.Variable):
    for n in graph.nodes:
        for k, inp in enumerate(n.inputs):
            if isinstance(inp, gs.Variable) and inp.name == old_var.name:
                n.inputs[k] = new_var
    for k, out in enumerate(graph.outputs):
        if isinstance(out, gs.Variable) and out.name == old_var.name:
            graph.outputs[k] = new_var


def _mark_drop(nodes_to_drop_ids: set, *nodes: gs.Node) -> None:
    for n in nodes:
        if n is not None:
            nodes_to_drop_ids.add(id(n))


def _detach_nodes(graph: gs.Graph, nodes_to_drop_ids: set) -> None:
    for n in list(graph.nodes):
        if id(n) not in nodes_to_drop_ids:
            continue
        for t in n.inputs:
            if hasattr(t, "outputs") and n in t.outputs:
                t.outputs.remove(n)
        for t in n.outputs:
            if hasattr(t, "inputs") and n in t.inputs:
                t.inputs.remove(n)
        n.inputs = []
        n.outputs = []


def _match_quant_from_var(var: gs.Variable, producers) -> tuple | None:
    """
    Match quant chain producing `var`:
      Div -> (optional Add zpshift) -> Round -> Clip
    """
    clip = producers.get(var.name)
    if clip is None or clip.op != "Clip" or not clip.inputs:
        return None

    rnd = producers.get(clip.inputs[0].name)
    if rnd is None or rnd.op != "Round" or not rnd.inputs:
        return None

    prev = producers.get(rnd.inputs[0].name)
    add = None
    if prev is not None and prev.op == "Add" and prev.inputs:
        add = prev
        prev = producers.get(add.inputs[0].name)

    div = prev
    if div is None or div.op != "Div":
        return None

    src = _non_const_input(div)
    if src is None:
        return None

    return div, add, rnd, clip, src


def _strip_qdq_backwards(var: gs.Variable, producers, nodes_to_drop_ids: set) -> gs.Variable:
    """
    Repeatedly strip trailing quant or dequant producers from var.
    """
    cur = var
    changed = True
    while changed:
        changed = False

        # Strip quant chain
        q = _match_quant_from_var(cur, producers)
        if q is not None:
            div, add, rnd, clip, src = q
            _mark_drop(nodes_to_drop_ids, div, add, rnd, clip)
            cur = src
            changed = True
            continue

        # Strip dequant
        p = producers.get(cur.name)
        if _is_mul_dequant(p):
            src = _non_const_input(p)
            if src is not None:
                _mark_drop(nodes_to_drop_ids, p)
                cur = src
                changed = True
                continue

    return cur


def _has_input_ancestor(var: gs.Variable, producers: dict, graph_input_names: set, max_hops: int = 256) -> bool:
    """
    True if `var` is on a path originating from a graph input.
    """
    cur = var
    hops = 0
    while isinstance(cur, gs.Variable) and hops < max_hops:
        if cur.name in graph_input_names:
            return True
        p = producers.get(cur.name)
        if p is None:
            return False
        nxt = _non_const_input(p)
        if nxt is None:
            return False
        cur = nxt
        hops += 1
    return False
# ...existing code...

def canonicalize_qdq_graph(
    model: onnx.ModelProto,
    assume_input_quantized: bool = True,
    remove_output_dequant: bool = True,
) -> onnx.ModelProto:
    graph = gs.import_onnx(model)
    nodes_to_drop_ids = set()

    # ---------- Rule 1: remove Q/DQ only on activation path to first Linear_OP ----------
    producers, consumers = _build_maps(graph)
    graph_input_names = {i.name for i in graph.inputs if isinstance(i, gs.Variable)}

    if assume_input_quantized:
        first_linear = next((n for n in graph.nodes if n.op in _LINEAR_OPS), None)
        if first_linear is not None and first_linear.inputs:
            # Only activation input (index 0), never weights/bias
            inp = first_linear.inputs[0]
            if isinstance(inp, gs.Variable) and _has_input_ancestor(inp, producers, graph_input_names):
                first_linear.inputs[0] = _strip_qdq_backwards(inp, producers, nodes_to_drop_ids)

    # ---------- Rule 3: no dequant before pooling ----------
    producers, consumers = _build_maps(graph)
    for pool in graph.nodes:
        if pool.op not in _POOL_OPS or not pool.inputs:
            continue

        in0 = pool.inputs[0]
        if not isinstance(in0, gs.Variable):
            continue

        p = producers.get(in0.name)
        if _is_mul_dequant(p):
            src = _non_const_input(p)
            if src is not None:
                pool.inputs[0] = src
                _mark_drop(nodes_to_drop_ids, p)

    # Also remove immediate quant right after pool: pool -> Div/(Add)->Round->Clip
    producers, consumers = _build_maps(graph)
    for pool in graph.nodes:
        if pool.op not in _POOL_OPS or not pool.outputs:
            continue
        pool_out = pool.outputs[0]
        pool_users = consumers.get(pool_out.name, [])
        if len(pool_users) != 1:
            continue
        u = pool_users[0]
        if u.op != "Div":
            continue

        div = u
        c1 = consumers.get(div.outputs[0].name, [])
        add = None
        if len(c1) == 1 and c1[0].op == "Add":
            add = c1[0]
            c1 = consumers.get(add.outputs[0].name, [])
        if len(c1) != 1 or c1[0].op != "Round":
            continue
        rnd = c1[0]
        c2 = consumers.get(rnd.outputs[0].name, [])
        if len(c2) != 1 or c2[0].op != "Clip":
            continue
        clip = c2[0]

        q_out = clip.outputs[0]
        _replace_all_consumers(graph, q_out, pool_out)
        _mark_drop(nodes_to_drop_ids, div, add, rnd, clip)

    # ---------- Generic collapse: DQ -> Q (restricted) ----------
    # Only collapse if resulting quantized tensor does NOT feed a linear op.
    producers, consumers = _build_maps(graph)
    for n in list(graph.nodes):
        if not _is_mul_dequant(n) or not n.outputs:
            continue

        dq_out = n.outputs[0]
        users = consumers.get(dq_out.name, [])
        if len(users) != 1:
            continue

        div = users[0]
        if div.op != "Div":
            continue

        c1 = consumers.get(div.outputs[0].name, [])
        add = None
        if len(c1) == 1 and c1[0].op == "Add":
            add = c1[0]
            c1 = consumers.get(add.outputs[0].name, [])
        if len(c1) != 1 or c1[0].op != "Round":
            continue
        rnd = c1[0]
        c2 = consumers.get(rnd.outputs[0].name, [])
        if len(c2) != 1 or c2[0].op != "Clip":
            continue
        clip = c2[0]

        q_users = consumers.get(clip.outputs[0].name, [])
        if any(u.op in _LINEAR_OPS for u in q_users):
            # Keep quantization feeding Conv/Gemm/MatMul
            continue

        src = _non_const_input(n)
        if src is None:
            continue

        _replace_all_consumers(graph, clip.outputs[0], src)
        _mark_drop(nodes_to_drop_ids, n, div, add, rnd, clip)

    # ---------- Rule 5: remove output dequant ----------
    if remove_output_dequant:
        producers, _ = _build_maps(graph)
        for i, out in enumerate(graph.outputs):
            if not isinstance(out, gs.Variable):
                continue
            p = producers.get(out.name)
            if _is_mul_dequant(p):
                src = _non_const_input(p)
                if src is not None:
                    graph.outputs[i] = src
                    _mark_drop(nodes_to_drop_ids, p)

    _detach_nodes(graph, nodes_to_drop_ids)
    graph.nodes = [n for n in graph.nodes if id(n) not in nodes_to_drop_ids]
    graph.cleanup().toposort()
    return gs.export_onnx(graph)