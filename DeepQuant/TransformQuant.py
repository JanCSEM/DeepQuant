# ...existing code...
import onnx
import onnx_graphsurgeon as gs
import numpy as np
import onnx.helper as helper

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

def replace_mul_with_dequant_and_quant_pattern(onnx_model: onnx.ModelProto) -> onnx.ModelProto:
    """
    Replace (mul) with custom Dequant, and (div+round+clip) with custom Quant nodes.
    The new Quant nodes will have 'n_levels' and 'signed' attributes inferred from
    the Clip node's parameters.
    """
    graph = gs.import_onnx(onnx_model)
    nodes_to_add = []
    nodes_to_remove = []

    for node in graph.nodes:
        # --- Detect Dequant pattern: mul(input, scale) ---
        if node.op == "Mul" and _is_const(node.inputs[1]):
            # Create a new Dequant node
            dequant_node = gs.Node(
                op="Dequant",
                name=node.name + "_dequant" if node.name else "Dequant",
                inputs=[node.inputs[0], node.inputs[1]],
                outputs=node.outputs
            )
            nodes_to_add.append(dequant_node)
            nodes_to_remove.append(node)

        # --- Detect Quant pattern: div -> round -> clip ---
        if node.op == "Div" and _is_const(node.inputs[1]):
            # Check if the output of Div is used ONLY by a Round node
            if not node.outputs or len(node.outputs[0].outputs) != 1 or node.outputs[0].outputs[0].op != "Round":
                continue
            round_node = node.outputs[0].outputs[0]

            # Check if the output of Round is used ONLY by a Clip node
            if not round_node.outputs or len(round_node.outputs[0].outputs) != 1 or round_node.outputs[0].outputs[0].op != "Clip":
                continue
            clip_node = round_node.outputs[0].outputs[0]

            # --- Infer Attributes from Clip node ---
            # The Clip node must have constant min and max values
            if len(clip_node.inputs) < 3 or not _is_const(clip_node.inputs[1]) or not _is_const(clip_node.inputs[2]):
                continue

            clip_min_val = clip_node.inputs[1].values.item()
            clip_max_val = clip_node.inputs[2].values.item()

            # Determine signedness and number of levels
            signed = bool(clip_min_val < 0)
            n_levels = int(clip_max_val - clip_min_val + 1)

            # Create a new Quant node with the inferred attributes
            quant_node = gs.Node(
                op="Quant",
                name=clip_node.name + "_quant" if clip_node.name else "Quant",
                # Inputs from Div, outputs from Clip
                inputs=[node.inputs[0], node.inputs[1]],
                outputs=clip_node.outputs,
                attrs={"n_levels": n_levels, "signed": signed}
            )
            nodes_to_add.append(quant_node)
            # Mark the entire pattern for removal
            nodes_to_remove.extend([node, round_node, clip_node])

    # Add new nodes and remove old ones
    graph.nodes.extend(nodes_to_add)
    for n in nodes_to_remove:
        # Disconnect the node from the graph completely before removal
        n.outputs.clear()
    graph.cleanup().toposort()

    return gs.export_onnx(graph)

def fuse_rescale_qdq(onnx_model: onnx.ModelProto) -> onnx.ModelProto:
    """
    Fuses consecutive Quant -> Dequant pairs.
    If a (Quant -> Dequant) pair is followed immediately by another
    (Quant -> Dequant) pair, the first pair is removed.
    This is done iteratively and safely to prevent graph corruption.
    """
    graph = gs.import_onnx(onnx_model)
    graph.fold_constants()

    # Use a loop to repeatedly apply the fusion until no more patterns can be found.
    # This is the safest way to perform complex graph transformations.
    while True:
        fusion_occured = False
        # Iterate over a copy of the nodes, as the graph will be modified.
        for node in list(graph.nodes):
            # --- Start pattern match: Find the first Quant node (Quant1) ---
            if node.op != "Quant":
                continue
            quant1_node = node

            # --- Find the first Dequant node (Dequant1) ---
            # It must be the *only* consumer of Quant1's output.
            if not quant1_node.outputs or len(quant1_node.outputs[0].outputs) != 1:
                continue
            dequant1_node = quant1_node.outputs[0].outputs[0]
            if dequant1_node.op != "Dequant":
                continue

            # --- Find the second Quant node (Quant2) ---
            # It must be the *only* consumer of Dequant1's output.
            if not dequant1_node.outputs or len(dequant1_node.outputs[0].outputs) != 1:
                continue
            quant2_node = dequant1_node.outputs[0].outputs[0]
            if quant2_node.op != "Quant":
                continue

            # --- Pattern Matched: Quant1 -> Dequant1 -> Quant2 ---
            # The full (Q1->D1->Q2->D2) pattern is not needed for this fusion.
            # We will remove the first pair (Quant1, Dequant1).

            # Reroute the graph: Connect the input of Quant1 directly to Quant2.
            # This bypasses and isolates the first Q->DQ pair.
            quant2_node.inputs[0] = quant1_node.inputs[0]

            # Mark the isolated nodes for removal.
            quant1_node.outputs.clear()
            dequant1_node.outputs.clear()

            fusion_occured = True
            # A fusion has changed the graph. Break the inner loop and
            # restart the scan from the beginning to ensure a consistent state.
            break

        # If a full pass completes with no fusions, the process is done.
        if not fusion_occured:
            break

    # After all fusions are complete, clean up the isolated nodes once.
    graph.cleanup().toposort()
    return gs.export_onnx(graph)

def remove_trailing_qdq(onnx_model: onnx.ModelProto) -> onnx.ModelProto:
    """
    Removes any Quant and/or Dequant nodes between the last computational
    node and the graph's output. This ensures the model output is in
    floating-point format. This version iteratively traces backwards from
    each output to handle any number of trailing QDQ nodes.
    """
    graph = gs.import_onnx(onnx_model)
    nodes_to_remove = []

    # Iterate backwards to safely modify the list of outputs
    for i in range(len(graph.outputs) - 1, -1, -1):
        current_tensor = graph.outputs[i]

        # Repeatedly trace backwards from the output tensor
        while True:
            # The tensor must be produced by exactly one node to be part of a chain
            if not current_tensor.inputs or len(current_tensor.inputs) != 1:
                break

            producer_node = current_tensor.inputs[0]

            # If the producer is a Quant or Dequant node, we can remove it
            if producer_node.op in ["Quant", "Dequant"]:
                # The new candidate tensor is the input to this QDQ node
                # We assume the first input is the data tensor
                current_tensor = producer_node.inputs[0]
                nodes_to_remove.append(producer_node)
            else:
                # We've hit a computational node (or something else), so we stop.
                break

        # After tracing back, update the graph output to point to the final tensor
        graph.outputs[i] = current_tensor

    # Isolate all marked nodes before the final cleanup.
    for node in nodes_to_remove:
        node.outputs.clear()

    graph.cleanup().toposort()
    return gs.export_onnx(graph)