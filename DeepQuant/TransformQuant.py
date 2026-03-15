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

            # --- Find the Dequant node ---
            if not linear_op_node.outputs or len(linear_op_node.outputs[0].outputs) != 1:
                continue
            dequant_node = linear_op_node.outputs[0].outputs[0]
            if dequant_node.op != "Dequant" or dequant_node.domain != "ai.onnx.contrib":
                continue

            # --- Find the Quant node ---
            if not dequant_node.outputs or len(dequant_node.outputs[0].outputs) != 1:
                continue
            quant_node = dequant_node.outputs[0].outputs[0]
            if quant_node.op != "Quant" or quant_node.domain != "ai.onnx.contrib":
                continue

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
                log2D = np.int64(31 - emax)
                mul64 = np.round(effective_scale * (2.0 ** log2D)).astype(np.int64)

                # Renormalize if any overflow (>= 2^31)
                while np.any(mul64 >= (1 << 31)):
                    mul64 >>= 1
                    log2D -= 1

                mul = mul64.astype(np.int32)

                    
                # debug prints
                print(F"effective_scale: {effective_scale}")
                print(F"log2D: {log2D}")
                print(F"mul: {mul}")
                print(f"max_exponent: {emax}")
            else:
                # Scalar case (original logic)
                significand, exponent = np.frexp(effective_scale)
                mul = np.round(significand * (2**31)).astype(np.int32)
                log2D = 31 - exponent
            # squeeze mul
            if mul.shape and mul.shape[0] == 1:
                mul = np.squeeze(mul, axis=0)
                print(F"mul squeezed to : {mul}")
            # Remove bias from Conv and put it here:
            add = np.zeros_like(mul, dtype=np.int32)
            if len(linear_op_node.inputs) > 2 and isinstance(linear_op_node.inputs[2], gs.Constant):
                bias = linear_op_node.inputs[2].values.astype(np.float32)

                add = np.round(bias.reshape(add.shape) * mul / 2.0**log2D).astype(np.int32)
            
                print(F"op: {linear_op_node.op}, mul shape: {mul.shape}")
                if linear_op_node.op == "Conv" and add.ndim == 1:
                    # Get the number of spatial dimensions from the 'kernel_shape' attribute
                    spatial_dims = len(linear_op_node.inputs[1].shape) - 2  # weight shape is (out_channels, in_channels, *kernel_shape)
                    new_shape = [add.shape[0]] + [1] * spatial_dims
                    add = add.reshape(new_shape)
                linear_op_node.inputs.pop(2)

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
                    gs.Constant(f"{linear_op_node.name}_mul", np.array(mul, dtype=np.float32)),
                    gs.Constant(f"{linear_op_node.name}_add", np.array(add, dtype=np.float32))
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
            sub_out = gs.Variable(name=f"{node.name}_sub_out")
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
            div_output = gs.Variable(name=f"{node.name}_div_out")
            div_node = gs.Node(
                op="Div",
                name=f"{node.name}_quant_div",
                inputs=[data_input, scale_input],
                outputs=[div_output]
            )

            add_output = gs.Variable(name=f"{node.name}_add_out")
            zp = gs.Constant(name=f"{node.name}_zero_point", values=np.array(0, dtype=np.float32))
            add_node = gs.Node(
                op="Add",
                name=f"{node.name}_quant_add",
                inputs=[div_output, zp],  # Inputs should be [data, zero_point]
                outputs=[add_output],

            )

            round_output = gs.Variable(name=f"{node.name}_round_out")
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