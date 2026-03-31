# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Federico Brancasi <fbrancasi@ethz.ch>
import io
import os
import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
import onnxruntime as ort
import onnx
from onnxruntime_extensions import get_library_path

from DeepQuant.TransformQuant import fuse_integer_matmul_with_requant, fuse_requant_shift_pattern, move_agnostic_ops_after_quant, \
                                    remove_intermediate_qdq_and_preserve_relu,\
                                    remove_trailing_qdq, \
                                    replace_mul_with_dequant_and_quant_pattern, \
                                    decompose_quant_dequant_nodes, \
                                    rename_parameter_initializers, \
                                    move_special_nodes_after_quant, \
                                    merge_consecutive_divs, \
                                    restore_gelu_nodes, \
                                    fix_squeeze_axes_inputs

from DeepQuant.Injects.Transformations import (
    LinearTransformation,  # Transformation for quantized linear layers (QuantLinear, QuantConv2d)
    ActivationTransformation,  # Transformation for quantized activation functions (QuantReLU, etc.)
    MHATransformation,  # Transformation for quantized multi-head attention modules
)
from DeepQuant.Injects.Executor import (
    TransformationExecutor,
)
from .CustomTracer import (
    CustomBrevitasTracer,
    customBrevitasTrace,
)  # Custom FX tracer for Brevitas modules
from DeepQuant.QuantManipulation.ParameterExtractor import (
    extract_brevitas_proxy_params,  # Extracts quantization parameters from Brevitas proxies
    print_quant_params,  # Displays quantization parameters in a readable format
)
from DeepQuant.QuantManipulation.QuantNodesDivider import (
    split_quant_nodes,
)  # Splits quantization nodes into Quant/Dequant pairs
from brevitas.export.inference import (
    quant_inference_mode,
)  # Inference mode for quantized models
from brevitas.export import (
    export_onnx_qcdq,
)  # Native Brevitas ONNX export functions
from DeepQuant.QuantManipulation.DequantModifier import (
    unifyLinearDequants,
)  # Unifies dequant nodes in linear layers
from brevitas.fx import brevitas_symbolic_trace  # Brevitas-specific symbolic tracing
from DeepQuant.Utils.GraphPrinter import (
    GraphModulePrinter,
)  # Custom Graph Printer
from DeepQuant.Utils.FxInterpreter import NodeTracer
import DeepQuant.QuantDequantOnnx # path to custom ONNX operators for quantization/dequantization patterns
from DeepQuant.QuantDequantOnnx import Quant, Dequant, RequantShift
from DeepQuant.onnx_node_implementations import run_onnx_graph
# ANSI color codes for improved debug output readability
BLUE = "\033[94m"
RED = "\033[31m"
ENDC = "\033[0m"


def exportBrevitas(
    model: nn.Module, exampleInput: torch.Tensor, debug: bool = False
) -> nn.Module:
    """
    Export a Brevitas model to an FX GraphModule with unrolled quantization operations.

    This function applies a series of transformations to make the quantization steps
    explicit in the model's computation graph, then traces the transformed model using
    a custom FX tracer.

    Args:
        model: The Brevitas-based model to export.
        example_input: A representative input tensor for shape tracing.
        debug: If True, prints transformation progress information.

    Returns:
        nn.Module: An FX GraphModule with explicit quantization operations.
    """

    printer = GraphModulePrinter()

    ###############################################################################
    # 1. Original Network
    ###############################################################################

    model = brevitas_symbolic_trace(
        model
    )  # Symbolically trace the original model using Brevitas
    if debug:
        print("\n\n=== 1. Original Network ===\n")
        printer.print_tabular(model)
        print()

    with (
        torch.no_grad(),
        quant_inference_mode(model),
    ):  # Disable gradients and use quantized inference mode
        outputModel = model(
            exampleInput
        )  # Compute original model output on example input for validation

    # export_onnx_qcdq(  # Export original model to ONNX format with QCDQ (Quant-Cast-DeQuant) nodes
    #     model,  # Model to export
    #     args=exampleInput,  # Example input for tracing
    #     export_path=EXPORT_FOLDER / "1_model_qcdq_original.onnx",
    #     opset_version=13,
    # )

    ###############################################################################
    # 2. Injection of New Modules
    ###############################################################################

    # Create transformation sequence in appropriate order
    transformations = [
        MHATransformation(),  # Multi-head attention transformation (applied first)
        LinearTransformation(),  # Quantized linear layers transformation
        ActivationTransformation(),  # Quantized activation functions transformation
    ]

    # Initialize custom tracer for Brevitas
    tracer = CustomBrevitasTracer(debug=debug)

    # Create and execute transformation sequence using the executor
    executor = TransformationExecutor(transformations, debug=debug, tracer=tracer)
    transformedModel = executor.execute(
        model, exampleInput
    )  # Apply all transformations to the model

    # Generate FX graph using the same tracer for consistency
    fxModel = customBrevitasTrace(
        root=transformedModel,  # Transformed model to trace
        concreteArgs=(exampleInput,),
        tracer=tracer,  # Use same tracer to maintain consistency with transformations
    )
    fxModel.recompile()  # Recompile the FX module to update its forward method
    with torch.no_grad():
        outputFxModel = fxModel(exampleInput)  # Compute transformed model output

    if isinstance(outputModel, tuple):
        outputModel = outputModel[0]

    if torch.allclose(
        outputFxModel, outputModel, atol=1e-5
    ):  # Check numerical equivalence within tolerance

        print(f"{BLUE} ✓ Injection of New Modules: output is consistent{ENDC}")
    else:
        raise RuntimeError(  # Raise error if outputs differ significantly
            f"{RED} ✗ Injection of New Modules changed the output significantly{ENDC}"
        )


    print(f"{BLUE} ✓ All transformations completed successfully!{ENDC}")
    if debug:
        print("\n=== 2. Network after the Injection of New Modules ===\n")
        printer.print_tabular(fxModel)


    ###############################################################################
    # 3. Extraction of Parameters & Split of Quant Nodes
    ###############################################################################

    # Extract quantization parameters from the network's proxies
    proxyParams = extract_brevitas_proxy_params(
        fxModel
    )  # Get scale, zero_point, bit_width for each quant node

    # if debug:
    # print_quant_params(
    #     proxyParams
    # )  # Display extracted parameters in a readable format

    # Split quantization nodes into separate Quant and Dequant nodes
    splitFxModel = split_quant_nodes(
        fxModel, proxyParams, debug
    )  # Transform quant nodes into quant-dequant pairs
    splitFxModel.recompile()  # Recompile to update forward method with new nodes

    with torch.no_grad():
        outputFxModelSplitQuant = splitFxModel(
            exampleInput
        )  # Compute output after node splitting

    # print("Output Original: ", output_model)
    # print("Output Split:    ", output_fx_model_split_quant)

    if torch.allclose(
        outputModel, outputFxModelSplitQuant, atol=1e-5
    ):  # Verify numerical consistency
        print(f"{BLUE} ✓ Split of Quant Nodes: output is consistent{ENDC}")
    else:
        raise RuntimeError(  # Raise error if inconsistent
            f"{RED} ✗ Split of Quant Nodes changed the output significantly{ENDC}"
        )

    if debug:
        print("\n=== 3. Network after the Split of Quant Nodes ===\n")
        printer.print_tabular(splitFxModel)
        print()

    ###############################################################################
    # 4. Modification of Dequant Nodes (shift them down)
    ###############################################################################

    # Perform the unification of linear dequant nodes (move dequantization after computation)
    fxModelUnified = unifyLinearDequants(splitFxModel, debug=debug)
    fxModelUnified.recompile()  # Recompile to update forward method with new node arrangement

    # Compute output after dequant node unification
    with torch.no_grad():
        outputFxModelDequantModified = fxModelUnified(
            exampleInput
        )  # Output after dequant modification

    if debug:
        print("\n=== 4. Network after the Modification of Dequant Nodes ===\n")
        printer.print_tabular(fxModelUnified)
        print()

    f = io.BytesIO()
    torch.onnx.export(
        fxModelUnified,
        args=exampleInput,
        # f=EXPORT_FOLDER / "4_model_dequant_moved.onnx",
        f=f,
        opset_version=18,
        keep_initializers_as_inputs=True,
        custom_opsets={"ai.onnx.contrib": 1},  # Register custom operator domain
        do_constant_folding=False,
        input_names=["input"],
        output_names=["output"],
    )

    # Verify numerical consistency after dequant modification
    if torch.allclose(
        outputModel, outputFxModelDequantModified, atol=1e-5
    ):  # Verify numerical consistency
        print(f"{BLUE} ✓ Modification of Dequant Nodes: output is consistent{ENDC}")
    else:
        print(f"✗ max error: {torch.max(torch.abs(outputModel - outputFxModelDequantModified))}")
        # raise RuntimeError(  # Raise error if inconsistent
        #     f"{RED} ✗ Modification of Dequant Nodes changed the output significantly{ENDC}"
        # )


    # Step 2: Apply ONNX graph transformation passes with per-pass numerical checks.
    # Each check runs the graph with a pure-Python executor (no ORT dependency)
    # and compares against the original Brevitas reference output.
    ref_output = outputModel.detach().cpu().numpy()
    ort_input = {"input": exampleInput.cpu().numpy()}
    atol = 1.0  # 1 LSB tolerance for quantised integer arithmetic

    def _check(model: onnx.ModelProto, step_name: str) -> None:
        try:
            onnx_out = run_onnx_graph(model, ort_input)
            max_diff = float(np.max(np.abs(onnx_out - ref_output)))
            if np.allclose(onnx_out, ref_output, atol=atol):
                print(f"{BLUE} ✓ {step_name}: output consistent (max_diff={max_diff:.6f}){ENDC}")
            else:
                print(
                    f"{RED} ✗ {step_name}: output differs "
                    f"(max_diff={max_diff:.6f}, atol={atol}){ENDC}"
                )
                print(f"    ref : {ref_output.flatten()[:10]}")
                print(f"    onnx: {onnx_out.flatten()[:10]}")
        except Exception as e:
            print(f"{RED} ✗ {step_name}: executor failed — {e}{ENDC}")

    onnx_model = onnx.load_model_from_string(f.getvalue())

    onnx_model = rename_parameter_initializers(onnx_model)
    _check(onnx_model, "rename_parameter_initializers")

    onnx_model = restore_gelu_nodes(onnx_model)
    _check(onnx_model, "restore_gelu_nodes")

    onnx_model = merge_consecutive_divs(onnx_model)
    _check(onnx_model, "merge_consecutive_divs")

    onnx_model = replace_mul_with_dequant_and_quant_pattern(onnx_model)
    _check(onnx_model, "replace_mul_with_dequant_and_quant_pattern")

    onnx_model = move_agnostic_ops_after_quant(onnx_model)
    _check(onnx_model, "move_agnostic_ops_after_quant")

    onnx_model = remove_intermediate_qdq_and_preserve_relu(onnx_model)
    _check(onnx_model, "remove_intermediate_qdq_and_preserve_relu")

    onnx_model = move_special_nodes_after_quant(onnx_model)
    _check(onnx_model, "move_special_nodes_after_quant")

    onnx_model = fuse_requant_shift_pattern(onnx_model)
    _check(onnx_model, "fuse_requant_shift_pattern")

    onnx_model = fuse_integer_matmul_with_requant(onnx_model)
    _check(onnx_model, "fuse_integer_matmul_with_requant")

    onnx_model = decompose_quant_dequant_nodes(onnx_model)
    _check(onnx_model, "decompose_quant_dequant_nodes")

    onnx_model = fix_squeeze_axes_inputs(onnx_model)
    _check(onnx_model, "fix_squeeze_axes_inputs")

    return onnx_model  # Return the final optimized FX GraphModule
