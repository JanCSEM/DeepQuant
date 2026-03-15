from matplotlib import scale
import numpy as np
from onnxruntime_extensions import onnx_op, PyOp

@onnx_op(op_type="QuantWeight", inputs=[PyOp.dt_float, PyOp.dt_float], outputs=[PyOp.dt_float])
def quant_weight_onnx(w, scale, n_levels=256, signed=True):
    """
    Quantize weights for ONNX export.

    Args:
        w: Weight tensor (numpy array)
        scale: Quantization scale
        zero_point: Quantization zero point
        dtype: Target data type (default int8  for weights)
    Returns:
        Quantized weight tensor as numpy array
    """
    signed = bool(signed)
    if signed and n_levels == 256:
        qmin, qmax = -2**7+1, 2**7-1
        q_w = np.clip(np.round(w / scale), qmin, qmax).astype(np.int8)

    elif not signed and n_levels == 256:
        qmin, qmax = 0, 2**8-1
        q_w = np.clip(np.round(w / scale), qmin, qmax).astype(np.uint8)

    elif signed and n_levels == 2**32:
        qmin, qmax = -2**31+1, 2**31-1
        q_w = np.clip(np.round(w / scale), qmin, qmax).astype(np.int32)

    elif not signed and n_levels == 2**32:
        qmin, qmax = 0, 2**32-1
        q_w = np.clip(np.round(w / scale), qmin, qmax).astype(np.uint32)
    else:
        raise ValueError(f"Unsupported combination of signed={signed} and n_levels={n_levels}")
    return q_w

@onnx_op(op_type="Quant", inputs=[PyOp.dt_float,  PyOp.dt_float, PyOp.dt_int64, 
                                  PyOp.dt_int64], 
                            outputs=[PyOp.dt_float])
def quant_activation_onnx(x, scale, n_levels, signed, zero_point=0.0):
    """
    Quantize weights for ONNX export.

    Args:
        w: Weight tensor (numpy array)
        scale: Quantization scale
        zero_point: Quantization zero point
        dtype: Target data type (default int8  for weights)
    Returns:
        Quantized weight tensor as numpy array
    """
    
    signed = bool(signed)
    if signed and n_levels == 256:
        qmin, qmax = -2**7, 2**7-1
        q_w = np.clip(np.round(x / scale + zero_point), qmin, qmax).astype(np.int8)

    elif not signed and n_levels == 256:
        qmin, qmax = 0, 2**8-1
        q_w = np.clip(np.round(x / scale + zero_point), qmin, qmax).astype(np.uint8)
    elif signed and n_levels == 2**32:
        qmin, qmax = -2**31, 2**31-1
        q_w = np.clip(np.round(x / scale + zero_point), qmin, qmax).astype(np.int32)

    elif not signed and n_levels == 2**32:
        qmin, qmax = 0, 2**32-1
        q_w = np.clip(np.round(x / scale + zero_point), qmin, qmax).astype(np.uint32)
    
    # special pass for fused ReLU
    elif signed and n_levels ==128:
        qmin, qmax = 0, 2**7-1
        q_w = np.clip(np.round(x / scale + zero_point), qmin, qmax).astype(np.uint8)
    else:
        raise ValueError(f"Unsupported combination of signed={signed} and n_levels={n_levels}")
    return q_w


@onnx_op(op_type="Dequant", inputs=[PyOp.dt_float, PyOp.dt_float],
                            outputs=[PyOp.dt_float])
def dequant_onnx(q_x, scale, zero_point=0.0):
    """
    Dequantize tensor for ONNX export.

    Args:
        q_x: Quantized tensor (numpy array)
        scale: Quantization scale
        zero_point: Quantization zero point
    Returns:
        Dequantized tensor as numpy array
    """    
    return (q_x.astype(np.float32) - zero_point) * scale



@onnx_op(op_type="RequantShift", inputs=[PyOp.dt_float, PyOp.dt_float, PyOp.dt_float, PyOp.dt_float, PyOp.dt_int64, PyOp.dt_int64, PyOp.dt_bool],
                            outputs=[PyOp.dt_float])
def requant_shift_onnx(q_x, mul, add, div, qmin, qmax, signed):

    input_offset = 0
    output_offset = 0
    rounding = 1
    log2D = int(np.log2(div))
    intermediate = q_x + input_offset * mul + add
    intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) + output_offset
    out = np.clip(intermediate, qmin, qmax)

    return out.astype(q_x.dtype)