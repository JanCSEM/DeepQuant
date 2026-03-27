# ...existing code...
import math
import onnx
import onnx_graphsurgeon as gs
import numpy as np
import onnx.helper as helper

_LINEAR_OPS = {"Conv", "Gemm", "MatMul"}
_QUANT_AGNOSTIC_OPS = {"MaxPool", "Reshape", "Gather", "Flatten", "AveragePool", "GlobalAveragePool", "Transpose", "Squeeze", "Unsqueeze"}
_NONLINEAR_OPS = {"Gelu", "Softmax", "Sigmoid", "Tanh"}
_NORM_OPS = {"BatchNorm", "LayerNorm", "GroupNorm"}
_PARAMETERIZABLE_OPS = _LINEAR_OPS.union(_NORM_OPS)

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

def rename_parameter_initializers(onnx_model: onnx.ModelProto) -> onnx.ModelProto:
    """
    Renames the initializers (weights and biases) of parameterizable layers
    to match PyTorch's naming convention (e.g., 'node_name.weight').
    """
    graph = gs.import_onnx(onnx_model)

    for node in graph.nodes:
        if node.op not in _PARAMETERIZABLE_OPS:
            continue

        # Node name must not be empty to create a meaningful name
        if not node.name:
            continue

        # --- Handle Weight Tensor (usually the second input) ---
        if len(node.inputs) > 1 and _is_const(node.inputs[1]):
            weight_tensor = node.inputs[1]
            weight_tensor.name = f"{node.name}.weight"

        # --- Handle Bias Tensor (usually the third input) ---
        if len(node.inputs) > 2 and _is_const(node.inputs[2]):
            bias_tensor = node.inputs[2]
            # For LayerNorm/GroupNorm, the third input is the bias.
            # For Conv/Gemm, it's also the bias.
            bias_tensor.name = f"{node.name}.bias"

    return gs.export_onnx(graph)

def move_agnostic_ops_after_quant(onnx_model: onnx.ModelProto) -> onnx.ModelProto:
    """
    Moves quantization-agnostic operations to be after a QDQ pair.
    Identifies the pattern: Dequant -> AgnosticOp -> Quant
    And transforms it to: Dequant -> Quant -> AgnosticOp
    This allows the agnostic operation to be performed on integer data.
    """
    graph = gs.import_onnx(onnx_model)
    graph.fold_constants()

    while True:
        fusion_occured = False
        for node in list(graph.nodes):
            # --- Start pattern match: Find a Dequant node ---
            if node.op != "Dequant":
                continue
            dequant_node = node

            # --- Find the AgnosticOp node ---
            # It must be the *only* consumer of the Dequant node's output.
            if not dequant_node.outputs or len(dequant_node.outputs[0].outputs) != 1:
                continue
            agnostic_op_node = dequant_node.outputs[0].outputs[0]
            if agnostic_op_node.op not in _QUANT_AGNOSTIC_OPS:
                continue

            # --- Find the Quant node ---
            # It must be the *only* consumer of the AgnosticOp's output.
            if not agnostic_op_node.outputs or len(agnostic_op_node.outputs[0].outputs) != 1:
                continue
            quant_node = agnostic_op_node.outputs[0].outputs[0]
            if quant_node.op != "Quant":
                continue

            # --- Pattern Matched: Dequant -> AgnosticOp -> Quant ---
            # --- Reroute the graph ---
            # 1. The Quant node's data input should now be the Dequant's output.
            quant_node.inputs[0] = dequant_node.outputs[0]
            # 2. The AgnosticOp's data input should now be the Quant's output.
            agnostic_op_node.inputs[0] = quant_node.outputs[0]
            if not quant_node.outputs:
                pass
            else:
                # find all consumers of quant_node's output and reroute them to agnostic op's output
                for consumer in quant_node.outputs[0].outputs:
                    for k, inp in enumerate(consumer.inputs):
                        if inp == quant_node.outputs[0]:
                            consumer.inputs[k] = agnostic_op_node.outputs[0]


            fusion_occured = True
            # A fusion has changed the graph. Break and restart the scan.
            break

        if not fusion_occured:
            break

    graph.cleanup().toposort()
    return gs.export_onnx(graph)

def restore_gelu_nodes(onnx_model: onnx.ModelProto) -> onnx.ModelProto:
    """
    Detects the standard GeLU decomposition subgraph and replaces it with a single GeLU node.
    """
    graph = gs.import_onnx(onnx_model)
    graph.fold_constants()

    fusion_occured = True
    print(f"Starting GeLU restoration pass...")
    while fusion_occured:
        fusion_occured = False
        for node in list(graph.nodes):
            # Stage 1: Find Mul node with two variable inputs
            if node.op != "Mul":
                continue
            mul_inputs = node.inputs
            if len(mul_inputs) != 2 or not all(isinstance(inp, gs.Variable) for inp in mul_inputs):
                continue

            # Identify mul_x and mul_const (mul_const is output of Mul with constant 0.5)
            mul_x, mul_const = mul_inputs
            mul_const_node = mul_const.inputs[0] if mul_const.inputs and mul_const.inputs[0].op == "Mul" else None
            if not mul_const_node:
                continue

            # Check mul_const_node has constant 0.5 input and Add input
            if len(mul_const_node.inputs) != 2:
                continue
            if not any(isinstance(inp, gs.Constant) and np.allclose(inp.values, 0.5, atol=1e-6) for inp in mul_const_node.inputs):
                continue
            add_var = [inp for inp in mul_const_node.inputs if isinstance(inp, gs.Variable)][0]
            add_node = add_var.inputs[0] if add_var.inputs and add_var.inputs[0].op == "Add" else None
            if not any(isinstance(inp, gs.Constant) and np.allclose(inp.values, 1.0, atol=1e-6) for inp in add_node.inputs):
                continue
            if not add_node:
                continue
            # Add node: inputs are Erf output and constant 1.0
            if len(add_node.inputs) != 2:
                continue
            
            print(F"found add node in GeLu pattern")

            erf_var = [inp for inp in add_node.inputs if isinstance(inp, gs.Variable)][0]
            erf_node = erf_var.inputs[0]  if erf_var.inputs and erf_var.inputs[0].op == "Erf" else None
            if not erf_node:
                continue
            print(F"found erf node in GeLu pattern")

            # Erf node input: output of Div node
            div_var = [inp for inp in erf_node.inputs if isinstance(inp, gs.Variable)][0]
            div_node = div_var.inputs[0] if div_var.inputs and div_var.inputs[0].op == "Div" else None
            # Div node: input is mul_x and const
            if not div_node:
                continue
            if len(div_node.inputs) != 2:
                continue
            print(F"found div node in GeLu pattern")
            # Confirm mul_x is used throughout the pattern
            if not any(inp.name == mul_x.name for inp in div_node.inputs):
                continue

            # Pattern matched: Replace with GeLU node
            print(f"Restoring GeLU: replacing {node.name} with GeLU({mul_x.name})")
            gelu_node = gs.Node(
                op="Gelu",
                name=f"restored_gelu_{node.name}",
                inputs=[mul_x],
                outputs=node.outputs,
                domain="com.microsoft"
            )
            graph.nodes.append(gelu_node)
            # Disconnect all nodes in the matched GeLU subgraph to prevent re-matching
            for n in [node, mul_const_node, add_node, erf_node, div_node]:
                if n is not None:
                    n.outputs.clear()
            fusion_occured = True
            break

    graph.cleanup().toposort()
    return gs.export_onnx(graph)

def fix_squeeze_axes_inputs(onnx_model: onnx.ModelProto) -> onnx.ModelProto:
    """
    Converts Squeeze nodes with 'axes' attribute to use 'axes' as an input tensor (for opset >= 13).
    """
    graph = gs.import_onnx(onnx_model)
    changed = False

    for node in graph.nodes:
        if node.op == "Squeeze" and "axes" in node.attrs:
            axes = node.attrs.pop("axes")
            # Insert axes as a Constant input tensor
            axes_tensor = gs.Constant(name=f"{node.name}_axes", values=np.array(axes, dtype=np.int64))
            # If Squeeze already has 2 inputs, replace the second; else, append
            if len(node.inputs) == 1:
                node.inputs.append(axes_tensor)
            else:
                node.inputs[1] = axes_tensor
            changed = True

    if changed:
        graph.cleanup().toposort()
    return gs.export_onnx(graph)

def merge_consecutive_divs(onnx_model: onnx.ModelProto) -> onnx.ModelProto:
    """
    Detects two consecutive Div nodes with constant divisors and merges them into one Div node
    with the divisor equal to the product of both initializers.
    """
    graph = gs.import_onnx(onnx_model)
    graph.fold_constants()

    fusion_occured = True
    while fusion_occured:
        fusion_occured = False
        for node in list(graph.nodes):
            if node.op != "Div":
                continue
            # Check if output feeds into another Div node
            if not node.outputs or len(node.outputs[0].outputs) != 1:
                continue
            next_node = node.outputs[0].outputs[0]
            if next_node.op != "Div":
                continue
            # Both divisors must be constants
            if len(node.inputs) < 2 or len(next_node.inputs) < 2:
                continue
            divisor1 = node.inputs[1]
            divisor2 = next_node.inputs[1]
            if not isinstance(divisor1, gs.Constant) or not isinstance(divisor2, gs.Constant):
                continue
            # Merge: new divisor is product of both
            merged_divisor = gs.Constant(
                name=f"{node.name}_merged_divisor",
                values=np.array(divisor1.values * divisor2.values)
            )
            # Create new Div node
            merged_div_node = gs.Node(
                op="Div",
                name=f"{node.name}_merged",
                inputs=[node.inputs[0], merged_divisor],
                outputs=next_node.outputs
            )
            graph.nodes.append(merged_div_node)
            # Disconnect old nodes
            node.outputs.clear()
            next_node.outputs.clear()
            fusion_occured = True
            break

    graph.cleanup().toposort()
    return gs.export_onnx(graph)

def move_special_nodes_after_quant(onnx_model: onnx.ModelProto) -> onnx.ModelProto:
    """
    Moves Reshape and Transpose nodes that are between Dequant and Quant to after the Quant node.
    Pattern: Dequant -> (Reshape|Transpose)* -> Quant
    Transforms to: Dequant -> Quant -> (Reshape|Transpose)*
    """
    graph = gs.import_onnx(onnx_model)
    graph.fold_constants()

    while True:
        fusion_occured = False
        for node in list(graph.nodes):
            # --- Start pattern match: Find a linear op (Conv, Gemm, MatMul) ---
            if node.op not in _LINEAR_OPS:
                continue
            linear_op_node = node
            add_node = None
            
            # --- New Transformer Pattern: MatMul -> Add ---
            # If we find a MatMul, check if its single consumer is an Add node.
            if node.op == "MatMul" and node.outputs and len(node.outputs[0].outputs) == 1:
                maybe_add_node = node.outputs[0].outputs[0]
                # The Add node must have a constant bias term.
                if maybe_add_node.op == "Add" and len(maybe_add_node.inputs) > 1 and _is_const(maybe_add_node.inputs[1]):
                    add_node = maybe_add_node
                    print(f"Found MatMul->Add pattern: {linear_op_node.name} -> {add_node.name}")

            # The node that feeds into the Dequant node is either the Add node or the original linear op.
            effective_linear_op = add_node if add_node else linear_op_node
            
            if not effective_linear_op.outputs or len(effective_linear_op.outputs[0].outputs) != 1:
                continue
            
            # -- Find the Dequant node
            dequant_node = effective_linear_op.outputs[0].outputs[0]
            if dequant_node.op != "Dequant" or dequant_node.domain != "ai.onnx.contrib":
                continue

            # Traverse through Reshape/Transpose nodes
            special_nodes = []
            next_node = dequant_node
            reroute = False
            while True:
                if not next_node.outputs or len(next_node.outputs[0].outputs) != 1:
                    break
                candidate = next_node.outputs[0].outputs[0]
                
                if candidate.op in {"Reshape", "Transpose"}:

                    special_nodes.append(candidate)
                    next_node = candidate
                    reroute = True
                else:
                    break

            if reroute:
                print(F"rerouting: {effective_linear_op.name} -> {dequant_node.name} -> {[n.name for n in special_nodes]} -> Quant")
                # Find Quant node after the last special node
                quant_node = next_node.outputs[0].outputs[0] if next_node.outputs and len(next_node.outputs[0].outputs) == 1 else None
                if not quant_node or quant_node.op != "Quant":
                    continue
                
                # Pattern matched: Dequant -> (Reshape|Transpose)* -> Quant
                # Move special nodes after Quant

                # 1. Quant node's input should be Dequant's output
                quant_node.inputs[0] = dequant_node.outputs[0]

                # 2. Each special node's input should be Quant's output (for the first), or previous special node's output
                prev_output = quant_node.outputs[0]
                for special_node in special_nodes:
                    special_node.inputs[0] = prev_output
                    prev_output = special_node.outputs[0]

                # 3. Reroute all consumers of Quant's output to the first special node's output (if any special nodes)
                if special_nodes:
                    for consumer in quant_node.outputs[0].outputs:
                        for k, inp in enumerate(consumer.inputs):
                            if inp == quant_node.outputs[0]:
                                consumer.inputs[k] = special_nodes[-1].outputs[0]

                # check new chain:
                print(F"new chain: {effective_linear_op.name} -> {dequant_node.name} -> {quant_node.name} -> {[n.name for n in special_nodes]} -> consumers: {[c.name for c in quant_node.outputs[0].outputs]}")
                fusion_occured = True
                break

        if not fusion_occured:
            break

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
                name=node.name + "_dequant" if node.name else "_dequant",
                inputs=[node.inputs[0], node.inputs[1]],
                outputs=node.outputs,
                domain="ai.onnx.contrib"
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
            bitwidth = int(math.log2(n_levels)) if n_levels > 0 else 0
            # Create a new Quant node with the inferred attributes
            quant_node = gs.Node(
                op="Quant",
                name=clip_node.name + "_quant" if clip_node.name else "_quant",
                # Inputs from Div, outputs from Clip
                inputs=[node.inputs[0],
                        node.inputs[1], # scale
                        gs.Constant(f"{clip_node.name}_n_levels", np.array(n_levels, dtype=np.int64)),
                        gs.Constant(f"{clip_node.name}_signed", np.array(int(signed), dtype=np.int64))                ],
                outputs=clip_node.outputs,
                attrs={"bit_width": bitwidth, "signed": int(signed)},
                domain="ai.onnx.contrib"
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

def remove_intermediate_qdq_and_preserve_relu(onnx_model: onnx.ModelProto) -> onnx.ModelProto:
    """
    Removes an intermediate Quant -> Dequant pair, preserving the ReLU-like
    clipping if the removed Quant node was unsigned.
    Identifies: Dequant1 -> Quant2 -> Dequant2 -> Quant3
    If Quant2 is unsigned, it modifies Quant3 to clip at 0.
    Otherwise, it connects Dequant1 directly to Quant3.
    """
    graph = gs.import_onnx(onnx_model)
    graph.fold_constants()

    while True:
        fusion_occured = False
        for node in list(graph.nodes):
            # --- Start pattern match: Find Dequant1 ---
            if node.op != "Dequant":
                continue
            dequant1_node = node

            # --- Find Quant2 ---
            if not dequant1_node.outputs or len(dequant1_node.outputs[0].outputs) != 1:
                continue
            quant2_node = dequant1_node.outputs[0].outputs[0]
            if quant2_node.op != "Quant":
                continue

            # --- Find Dequant2 ---
            if not quant2_node.outputs or len(quant2_node.outputs[0].outputs) != 1:
                continue
            dequant2_node = quant2_node.outputs[0].outputs[0]
            if dequant2_node.op != "Dequant":
                continue

            # --- Find Quant3 ---
            if not dequant2_node.outputs or len(dequant2_node.outputs[0].outputs) != 1:
                continue
            quant3_node = dequant2_node.outputs[0].outputs[0]
            if quant3_node.op != "Quant":
                continue

            # --- Pattern Matched: DQ1 -> Q2 -> DQ2 -> Q3 ---

            # Check if Quant2 is unsigned (acting as a ReLU)
            # We assume the 'signed' parameter is the 4th input (index 3)
            is_quant2_unsigned = False
            if len(quant2_node.inputs) > 3 and isinstance(quant2_node.inputs[3], gs.Constant):
                if int(quant2_node.inputs[3].values) == 0:
                    is_quant2_unsigned = True

            # Reroute the graph by connecting Dequant1's output to Quant3's input
            quant3_node.inputs[0] = dequant1_node.outputs[0]

            if is_quant2_unsigned:
                # --- Modify Quant3 to perform the ReLU clip ---
                # By changing n_levels to 128 on a signed quantizer, we force the range to [0, 127]
                # We assume n_levels is the 3rd input (index 2)
                if len(quant3_node.inputs) > 2 and isinstance(quant3_node.inputs[2], gs.Constant):
                    n_levels_const = quant3_node.inputs[2]
                    quant3_node.inputs[2] = gs.Constant(n_levels_const.name, np.array(128, dtype=np.int64))


            # Mark the intermediate nodes for removal.
            quant2_node.outputs.clear()
            dequant2_node.outputs.clear()

            fusion_occured = True
            break

        if not fusion_occured:
            break

    graph.cleanup().toposort()
    return gs.export_onnx(graph)

def _find_producer_quant_node_recursively(var: gs.Variable, producers: dict) -> gs.Node | None:
    """
    Recursively searches backwards from a variable to find the producing Quant node,
    skipping over quantization-agnostic operations.
    """
    if not isinstance(var, gs.Variable) or var.name not in producers:
        return None

    producer_node = producers[var.name]

    if producer_node.op == "Quant":
        return producer_node

    else:
        return _find_producer_quant_node_recursively(producer_node.inputs[0], producers)


def simplify_quant_dequant_nodes(onnx_model: onnx.ModelProto) -> onnx.ModelProto:
    """
    Simplifies custom Quant and Dequant nodes by moving quantization parameters
    from inputs to node attributes. It also converts the 'n_levels' parameter
    to 'bitwidth'.
    """
    graph = gs.import_onnx(onnx_model)
    producers, _ = _build_maps(graph)

    for node in list(graph.nodes):
        if node.domain != "ai.onnx.contrib":
            continue

        # --- Simplify Dequant node ---
        if node.op == "Dequant":
            # Dequant(data, scale) -> Dequant(data) with scale and zero_point attributes
            if len(node.inputs) > 1 and isinstance(node.inputs[1], gs.Constant):
                scale_const = node.inputs[1]
                node.attrs["scale"] = scale_const
                node.attrs["zero_point"] = 0  # Add zero_point attribute

                # Recursively find the preceding Quant node to get n_levels and signed status
                producer_quant_node = _find_producer_quant_node_recursively(node.inputs[0], producers)

                if producer_quant_node:
                    bitwidth = producer_quant_node.attrs["bit_width"] if "bit_width" in producer_quant_node.attrs else None
                    signed = producer_quant_node.attrs["signed"] if "signed" in producer_quant_node.attrs else None

                    node.attrs["bit_width"] =  bitwidth
                    node.attrs["signed"] = signed

                # Keep only the data input
                node.inputs = [node.inputs[0]]
        # --- Simplify Quant node ---
        elif node.op == "Quant":
            # Quant(data, scale, n_levels, signed) -> Quant(data) with attributes
            if len(node.inputs) > 3:
                scale_const = node.inputs[1]
                n_levels_const = node.inputs[2]
                signed_const = node.inputs[3]

                if all(isinstance(c, gs.Constant) for c in [scale_const, n_levels_const, signed_const]):
                    # Move scale and signed to attributes
                    node.attrs["scale"] = scale_const
                    node.attrs["signed"] = bool(signed_const.values.item())
                    node.attrs["zero_point"] = 0 # Add zero_point attribute

                    # Convert n_levels to bitwidth and add as attribute
                    n_levels = int(n_levels_const.values.item())
                    bitwidth = int(np.log2(n_levels)) if n_levels > 0 else 0
                    node.attrs["bit_width"] = bitwidth

                    # Keep only the data input
                    node.inputs = [node.inputs[0]]

    graph.cleanup().toposort()
    return gs.export_onnx(graph)

def replace_matmul_add_by_gemm(onnx_model: onnx.ModelProto) -> onnx.ModelProto:
    """
    Finds the pattern MatMul -> Add and replaces it with a single Gemm node.
    This is a common pattern in transformer models for linear layers.
    """
    graph = gs.import_onnx(onnx_model)
    graph.fold_constants()

    while True:
        fusion_occured = False
        for node in list(graph.nodes):
            # --- Start pattern match: Find a MatMul node ---
            if node.op != "MatMul":
                continue

            # The MatMul output must have exactly one consumer.
            if not node.outputs or len(node.outputs[0].outputs) != 1:
                continue
            
            # --- Find the Add node ---
            add_node = node.outputs[0].outputs[0]
            if add_node.op != "Add":
                continue

            # --- Check for constant weights and biases ---
            # MatMul must have a constant weight tensor (input B)
            if len(node.inputs) < 2 or not _is_const(node.inputs[1]):
                continue
            
            # Add must have a constant bias tensor
            bias_input = None
            for inp in add_node.inputs:
                if _is_const(inp):
                    bias_input = inp
                    break
            if bias_input is None:
                continue

            # --- Pattern Matched: MatMul(A, B) -> Add(C, D) ---
            print(f"Fusing pattern: MatMul({node.name}) -> Add({add_node.name})")

            # --- Create the new Gemm node ---
            # Inputs: A (from MatMul), B (from MatMul), C (from Add)
            gemm_inputs = [node.inputs[0], node.inputs[1], bias_input]
            
            # Output: The original output of the Add node
            gemm_outputs = add_node.outputs
            
            gemm_node = gs.Node(
                op="Gemm",
                name=f"{node.name}_gemm",
                inputs=gemm_inputs,
                outputs=gemm_outputs
            )
            
            # Add the new node to the graph
            graph.nodes.append(gemm_node)

            # --- Clean up the old nodes ---
            # Disconnect the old nodes so they can be removed by cleanup
            node.outputs.clear()
            add_node.outputs.clear()

            fusion_occured = True
            # A fusion has changed the graph. Break and restart the scan.
            break

        if not fusion_occured:
            break

    # Remove the old, disconnected nodes and re-sort the graph
    graph.cleanup().toposort()
    return gs.export_onnx(graph)

def fuse_requant_shift_pattern(onnx_model: onnx.ModelProto) -> onnx.ModelProto:
    """
    Finds patterns like Conv/Gemm -> Dequant -> Quant and fuses the
    Dequant -> Quant part into a single, custom RequantShift node.
    """
    graph = gs.import_onnx(onnx_model)
    graph.fold_constants()

    while True:
        fusion_occured = False
        for node in list(graph.nodes):
            # --- Start pattern match: Find a linear op (Conv, Gemm, MatMul) ---
            if node.op not in _LINEAR_OPS:
                continue
            linear_op_node = node
            add_node = None
            
            # --- New Transformer Pattern: MatMul -> Add ---
            # If we find a MatMul, check if its single consumer is an Add node.
            if node.op == "MatMul" and node.outputs and len(node.outputs[0].outputs) == 1:
                maybe_add_node = node.outputs[0].outputs[0]
                # The Add node must have a constant bias term.
                if maybe_add_node.op == "Add" and len(maybe_add_node.inputs) > 1 and _is_const(maybe_add_node.inputs[1]):
                    add_node = maybe_add_node
                    print(f"Found MatMul->Add pattern: {linear_op_node.name} -> {add_node.name}")

            # The node that feeds into the Dequant node is either the Add node or the original linear op.
            effective_linear_op = add_node if add_node else linear_op_node
            
            print(f"Found linear op: {effective_linear_op.op}, name: {effective_linear_op.name}")            # --- Find the Dequant node ---
            if not effective_linear_op.outputs or len(effective_linear_op.outputs[0].outputs) != 1:
                continue
            
            # -- Find the Dequant node
            dequant_node = effective_linear_op.outputs[0].outputs[0]
            if dequant_node.op != "Dequant" or dequant_node.domain != "ai.onnx.contrib":
                continue
            print(f"Found dequant op: {dequant_node.op}, name: {dequant_node.name}")

            # --- Find the Quant node ---
            quant_node = dequant_node.outputs[0].outputs[0] if dequant_node.outputs and len(dequant_node.outputs[0].outputs) == 1 else None
            if not quant_node or quant_node.op != "Quant" or quant_node.domain != "ai.onnx.contrib":
                continue
            print(f"Found Quant op: {dequant_node.op}, name: {dequant_node.name}")

            # --- Pattern Matched: LinearOp -> Dequant -> Quant ---

            # --- Extract Parameters for RequantShift ---
            if len(dequant_node.inputs) < 2 or not isinstance(dequant_node.inputs[1], gs.Constant):
                continue
            dequant_scale = dequant_node.inputs[1].values

            if len(quant_node.inputs) < 2 or not isinstance(quant_node.inputs[1], gs.Constant):
                continue
            quant_scale = quant_node.inputs[1].values

            output_zp = 0
            if len(quant_node.inputs) > 3 and isinstance(quant_node.inputs[3], gs.Constant):
                 is_signed = bool(quant_node.inputs[3].values.item())
                 if not is_signed:
                     output_zp = 0

            # --- Calculate RequantShift parameters ---
            # Effective scale for requantization
            effective_scale = dequant_scale / quant_scale

            if effective_scale.ndim > 0:
                # Per-channel case: Find a single best shift for all channels.
                # A good heuristic is to use the exponent of the maximum scale value
                # to avoid overflow and preserve precision.
                                
                _, emax = np.frexp(np.max(effective_scale))
                log2D = np.int64(15 - emax)
                mul64 = np.round(effective_scale * (2.0 ** log2D)).astype(np.int64)

                # Renormalize if any overflow (>= 2^31)
                while np.any(mul64 >= (1 << 31)):
                    mul64 >>= 1
                    log2D -= 1

                mul = mul64.astype(np.int32)

                # debug prints
                # print(F"effective_scale: {effective_scale}")
                # print(F"log2D: {log2D}")
                # print(F"mul: {mul}")
                # print(f"max_exponent: {emax}")
            else:
                # Scalar case (original logic)
                significand, exponent = np.frexp(effective_scale)
                mul = np.round(significand * (2**15)).astype(np.int32)
                log2D = 15 - exponent
            # squeeze mul
            if mul.shape and mul.shape[0] == 1:
                mul = np.squeeze(mul, axis=0)
                
            # --- Handle Bias ---
            # Remove bias from Conv/Gemm or take it from the separate Add node.
            add = np.zeros_like(mul, dtype=np.int32)
            bias_source_node = add_node if add_node else linear_op_node
            
            # Check for bias on the source node (Conv, Gemm, or Add)
            if len(bias_source_node.inputs) > 1 and _is_const(bias_source_node.inputs[1 if add_node else 2]):
                bias_input_index = 1 if add_node else 2
                bias = bias_source_node.inputs[bias_input_index].values.astype(np.float32)
                
                # The bias is scaled by the requant multiplier 'mul'
                add = (mul * bias.reshape(add.shape)).astype(np.int32)
                
                # Remove the original bias input
                bias_source_node.inputs.pop(bias_input_index)

            # --- Create the new RequantShift node ---
            
            requant_input_var = linear_op_node.outputs[0]
            final_output_var = quant_node.outputs[0]
            final_output_var.shape = requant_input_var.shape
            final_output_var.dtype = requant_input_var.dtype
            requant_shift_node = gs.Node(
                op="RequantShift",
                name=f"{linear_op_node.name}_requant_shift",
                domain="ai.onnx.contrib",
                inputs=[
                    requant_input_var,
                    gs.Constant(f"{linear_op_node.name}_mul", np.array(np.squeeze(mul), dtype=np.float32)),
                    gs.Constant(f"{linear_op_node.name}_add", np.array(np.squeeze(add), dtype=np.float32))
                ],
                outputs=[final_output_var],
                attrs={
                    "n_levels": gs.Constant(f"{linear_op_node.name}_n_levels", np.array([2**int(quant_node.attrs.get("bit_width", 0))], dtype=np.float32)),
                    "signed": gs.Constant(f"{linear_op_node.name}_signed", np.array([quant_node.attrs.get("signed", 0)], dtype=np.float32)),
                    "div": gs.Constant(f"{linear_op_node.name}_div", np.array(2**int(log2D), dtype=np.float32))
                }
               
            )
            if linear_op_node.op == "Conv":
                if "auto_pad" in linear_op_node.attrs:
                    del linear_op_node.attrs["auto_pad"]
                linear_op_node.attrs["kernel_shape"] = linear_op_node.inputs[1].shape[2:]
            graph.nodes.append(requant_shift_node)

            dequant_node.outputs.clear()
            quant_node.outputs.clear()

            fusion_occured = True
            break

        if not fusion_occured:
            break

    graph.cleanup().toposort()
    return gs.export_onnx(graph)

def fuse_integer_matmul_with_requant(onnx_model: onnx.ModelProto) -> onnx.ModelProto:
    """
    Fuses patterns of quantized integer MatMul followed by Dequant into
    IntegerMatMul -> RequantShift -> Dequant.
    """
    graph = gs.import_onnx(onnx_model)
    graph.fold_constants()

    fusion_occured = True
    print(f"Starting fusion of Integer MatMul with RequantShift...")
    while fusion_occured:
        fusion_occured = False
        for node in list(graph.nodes):
            # 1. Find MatMul node with both inputs from Dequant
            if node.op != "MatMul":
                continue
            if len(node.inputs) != 2:
                continue
            dequant_a, dequant_b = node.inputs
            if not (isinstance(dequant_a, gs.Variable) and isinstance(dequant_b, gs.Variable)):
                continue
            if not (dequant_a.inputs and dequant_a.inputs[0].op == "Dequant"):
                continue
            if not (dequant_b.inputs and dequant_b.inputs[0].op == "Dequant"):
                continue
            dequant_node_a = dequant_a.inputs[0]
            dequant_node_b = dequant_b.inputs[0]

            print(F"found MatMul with Dequant inputs: {node.name}, {dequant_node_a.name}, {dequant_node_b.name}")
            # # 2. Check both Dequant nodes are fed by Quant nodes
            # if not (dequant_node_a.inputs and dequant_node_a.inputs[0].inputs and dequant_node_a.inputs[0].inputs[0].op == "Quant"):
            #     continue
            # if not (dequant_node_b.inputs and dequant_node_b.inputs[0].inputs and dequant_node_b.inputs[0].inputs[0].op == "Quant"):
            #     continue
            quant_node_a = dequant_node_a.inputs[0].inputs[0]
            quant_node_b = dequant_node_b.inputs[0].inputs[0]

            # 3. Replace with IntegerMatMul -> RequantShift -> Dequant
            # (You may need to adjust input/output dtypes and attributes as needed)
            int_matmul_out = gs.Variable(name=f"{node.name}_int_matmul_out", dtype=np.int32, shape=node.outputs[0].shape)
            int_matmul_node = gs.Node(
                op="MatMul",
                name=f"{node.name}_int",
                inputs=[quant_node_a.outputs[0], quant_node_b.outputs[0]],
                outputs=[int_matmul_out]
            )
            # Dummy RequantShift parameters (replace with correct scale/bias computation)
            
            scale_a = dequant_node_a.inputs[1].values if len(dequant_node_a.inputs) > 1 else None
            scale_b = dequant_node_b.inputs[1].values if len(dequant_node_b.inputs) > 1 else None
            if scale_a is None or scale_b is None:
                continue

            # Compute effective scale for requantization (tensor-wise)
            effective_scale = scale_a * scale_b
            
            _, emax = np.frexp(np.max(effective_scale))
            log2D = np.int64(15 - emax)
            mul64 = np.round(effective_scale * (2.0 ** log2D)).astype(np.int64)

            # Renormalize if any overflow (>= 2^31)
            while np.any(mul64 >= (1 << 31)):
                mul64 >>= 1
                log2D -= 1

            mul_val = mul64.astype(np.int32)
            shape_dequant_b = dequant_node_b.inputs[0].shape
            mul = gs.Constant(f"{node.name}_mul", np.array([mul_val]*shape_dequant_b[2], dtype=np.float32))
            add = gs.Constant(f"{node.name}_add", np.zeros_like([mul_val]*shape_dequant_b[2], dtype=np.float32))
            
            requant_out = gs.Variable(name=f"{node.name}_requant_out", dtype=np.int32, shape=int_matmul_out.shape)
            requant_node = gs.Node(
                op="RequantShift",
                name=f"{node.name}_requant",
                domain="ai.onnx.contrib",
                inputs=[int_matmul_out, mul, add],
                outputs=[requant_out],
                attrs={
                    "n_levels": gs.Constant(f"{node.name}__n_levels", np.array([2**int(8)], dtype=np.float32)),
                    "signed": gs.Constant(f"{node.name}_signed", np.array([1], dtype=np.float32)),
                    "div": gs.Constant(f"{node.name}_div", np.array(2**int(log2D), dtype=np.float32))
                }
            )
            # Dequant node
            scale = gs.Constant(f"{node.name}_scale", np.array([effective_scale], dtype=np.float32))
            dequant_out = gs.Variable(name=f"{node.name}_dequant_out", dtype=np.float32)
            dequant_node = gs.Node(
                op="Dequant",
                name=f"{node.name}_dequant",
                domain="ai.onnx.contrib",
                inputs=[requant_out, scale],
                outputs=node.outputs
            )

            graph.nodes.extend([int_matmul_node, requant_node, dequant_node])
            node.outputs.clear()
            node.outputs.clear()
            dequant_node_a.outputs.clear()
            dequant_node_b.outputs.clear()
            fusion_occured = True
            break

    graph.cleanup().toposort()
    return gs.export_onnx(graph)

def decompose_quant_dequant_nodes(onnx_model: onnx.ModelProto) -> onnx.ModelProto:
    """
    Decomposes custom Quant and Dequant nodes back into standard ONNX operators.
    - Dequant(data, scale, zp) -> Sub(data, zp) -> Mul(data, scale)
    - Quant(data, scale, zp, n_levels, signed) -> Div -> Add -> Round -> Clip
    """
    graph = gs.import_onnx(onnx_model)
    nodes_to_add = []
    nodes_to_remove = []

    for node in list(graph.nodes):
        if node.domain != "ai.onnx.contrib":
            continue

        # --- Decompose Dequant node ---
        if node.op == "Dequant":
            # Dequant(data, scale) -> Mul(data, scale)
            # define new output variable for the Mul node
            sub_out = gs.Variable(name=f"{node.name}_sub_out", dtype=onnx.TensorProto.FLOAT, 
                                  shape=node.inputs[0].shape)
            zp = gs.Constant(name=f"{node.name}_zero_point", values=np.array(0, dtype=np.float32))
            sub_node = gs.Node(
                op="Sub",
                name=f"{node.name}_dequant_sub",
                inputs=[node.inputs[0], zp],  # Inputs should be [data, zero_point]
                outputs=[sub_out],

            )
            mul_node = gs.Node(
                op="Mul",
                name=f"{node.name}_dequant_mul",
                inputs=[sub_out, node.inputs[1]],  # Assumes inputs are [data, scale]
                outputs=node.outputs
            )
            nodes_to_add.append(sub_node)
            nodes_to_add.append(mul_node)
            nodes_to_remove.append(node)

        # --- Decompose Quant node ---
        elif node.op == "Quant":
            # Quant(data, scale, n_levels, signed) -> Div -> Round -> Clip
            data_input = node.inputs[0]
            scale_input = node.inputs[1]
            n_levels_input = node.inputs[2]
            signed_input = node.inputs[3]

            # --- Calculate Clip min/max from n_levels and signed ---
            n_levels = int(n_levels_input.values.item())
            is_signed = bool(signed_input.values.item())

            if is_signed and n_levels > 128:
                clip_min = -n_levels // 2
                clip_max = n_levels // 2 - 1
            # special relu case
            elif is_signed and n_levels == 128:
                clip_min = 0
                clip_max = 127
            else:
                clip_min = 0
                clip_max = n_levels - 1

            # --- Create the new node chain ---
            div_output = gs.Variable(name=f"{node.name}_div_out", 
                                     dtype=onnx.TensorProto.FLOAT, shape=data_input.shape)
            div_node = gs.Node(
                op="Div",
                name=f"{node.name}_quant_div",
                inputs=[data_input, scale_input],
                outputs=[div_output]
            )

            add_output = gs.Variable(name=f"{node.name}_add_out", dtype=onnx.TensorProto.FLOAT,
                                     shape=data_input.shape)
            zp = gs.Constant(name=f"{node.name}_zero_point", values=np.array(0, dtype=np.float32))
            add_node = gs.Node(
                op="Add",
                name=f"{node.name}_quant_add",
                inputs=[div_output, zp],  # Inputs should be [data, zero_point]
                outputs=[add_output],

            )

            round_output = gs.Variable(name=f"{node.name}_round_out",
                                        dtype=onnx.TensorProto.FLOAT, shape=data_input.shape)
            round_node = gs.Node(
                op="Round",
                name=f"{node.name}_quant_round",
                inputs=[add_output],
                outputs=[round_output]
            )

            clip_min_const = gs.Constant(name=f"{node.name}_clip_min", values=np.array(clip_min, dtype=np.float32))
            clip_max_const = gs.Constant(name=f"{node.name}_clip_max", values=np.array(clip_max, dtype=np.float32))
            clip_node = gs.Node(
                op="Clip",
                name=f"{node.name}_quant_clip",
                inputs=[round_output, clip_min_const, clip_max_const],
                outputs=node.outputs  # Final output is the original Quant node's output
            )

            nodes_to_add.extend([div_node, add_node, round_node, clip_node])
            nodes_to_remove.append(node)

    # Add new nodes and remove old ones
    graph.nodes.extend(nodes_to_add)
    for n in nodes_to_remove:
        n.outputs.clear()
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