from matplotlib import scale
import numpy as np
from onnx_ir import Function
from onnxruntime_extensions import onnx_op, PyOp

# @onnx_op(op_type="QuantWeight", inputs=[PyOp.dt_float, PyOp.dt_float], outputs=[PyOp.dt_float])
# def quant_weight_onnx(w, scale, n_levels=256, signed=True):
#     """
#     Quantize weights for ONNX export.

#     Args:
#         w: Weight tensor (numpy array)
#         scale: Quantization scale
#         zero_point: Quantization zero point
#         dtype: Target data type (default int8  for weights)
#     Returns:
#         Quantized weight tensor as numpy array
#     """
#     signed = bool(signed)
#     if signed and n_levels == 256:
#         qmin, qmax = -2**7+1, 2**7-1
#         q_w = np.clip(np.round(w / scale), qmin, qmax).astype(np.int8)

#     elif not signed and n_levels == 256:
#         qmin, qmax = 0, 2**8-1
#         q_w = np.clip(np.round(w / scale), qmin, qmax).astype(np.uint8)

#     elif signed and n_levels == 2**32:
#         qmin, qmax = -2**31+1, 2**31-1
#         q_w = np.clip(np.round(w / scale), qmin, qmax).astype(np.int32)

#     elif not signed and n_levels == 2**32:
#         qmin, qmax = 0, 2**32-1
#         q_w = np.clip(np.round(w / scale), qmin, qmax).astype(np.uint32)
#     else:
#         raise ValueError(f"Unsupported combination of signed={signed} and n_levels={n_levels}")
#     return q_w

# @onnx_op(op_type="Quant", inputs=[PyOp.dt_float,  PyOp.dt_float, PyOp.dt_int64, 
#                                   PyOp.dt_int64], 
#                             outputs=[PyOp.dt_float])
# def quant_activation_onnx(x, scale, n_levels, signed, zero_point=0.0):
#     """
#     Quantize weights for ONNX export.

#     Args:
#         w: Weight tensor (numpy array)
#         scale: Quantization scale
#         zero_point: Quantization zero point
#         dtype: Target data type (default int8  for weights)
#     Returns:
#         Quantized weight tensor as numpy array
#     """
    
#     signed = bool(signed)
#     if signed and n_levels == 256:
#         qmin, qmax = -2**7, 2**7-1
#         q_w = np.clip(np.round(x / scale + zero_point), qmin, qmax).astype(np.int8)

#     elif not signed and n_levels == 256:
#         qmin, qmax = 0, 2**8-1
#         q_w = np.clip(np.round(x / scale + zero_point), qmin, qmax).astype(np.uint8)
#     elif signed and n_levels == 2**32:
#         qmin, qmax = -2**31, 2**31-1
#         q_w = np.clip(np.round(x / scale + zero_point), qmin, qmax).astype(np.int32)

#     elif not signed and n_levels == 2**32:
#         qmin, qmax = 0, 2**32-1
#         q_w = np.clip(np.round(x / scale + zero_point), qmin, qmax).astype(np.uint32)
    
#     # special pass for fused ReLU
#     elif signed and n_levels ==128:
#         qmin, qmax = 0, 2**7-1
#         q_w = np.clip(np.round(x / scale + zero_point), qmin, qmax).astype(np.uint8)
#     else:
#         raise ValueError(f"Unsupported combination of signed={signed} and n_levels={n_levels}")
#     return q_w


# @onnx_op(op_type="Dequant", inputs=[PyOp.dt_float, PyOp.dt_float],
#                             outputs=[PyOp.dt_float])
# def dequant_onnx(q_x, scale, zero_point=0.0):
#     """
#     Dequantize tensor for ONNX export.

#     Args:
#         q_x: Quantized tensor (numpy array)
#         scale: Quantization scale
#         zero_point: Quantization zero point
#     Returns:
#         Dequantized tensor as numpy array
#     """    
#     return (q_x.astype(np.float32) - zero_point) * scale



@onnx_op(op_type="RequantShift", domain="ai.onnx.contrib", inputs=[PyOp.dt_float, PyOp.dt_float, PyOp.dt_float],
                            outputs=[PyOp.dt_float])
def requant_shift_onnx(q_x, mul, add, div=2**15, qmin=-128, qmax=127, signed=True):

    input_offset = 0
    output_offset = 0
    rounding = 1
    log2D = int(np.log2(div))
    
    # broadcast mul and add to match q_x shape if necessary
    if mul.ndim == 1 and mul.shape[0] == q_x.shape[1]:  # per-channel case
        mul = mul.reshape(1, -1, *[1]*(q_x.ndim-2))
    if add.ndim == 1 and add.shape[0] == q_x.shape[1]:  # per-channel case
        add = add.reshape(1, -1, *[1]*(q_x.ndim-2))
    intermediate = q_x + input_offset * mul + add
    intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) // 2.0**log2D) + output_offset
    out = np.clip(intermediate, qmin, qmax)

    return out.astype(q_x.dtype)

# @onnx_op(op_type="Gelu", domain="com.microsoft", inputs=[PyOp.dt_float],
#                             outputs=[PyOp.dt_float])
# def gelu_onnx(x):
#     """GELU activation function for ONNX export.
#     """
#     return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))

class RequantShift(Function):
    @staticmethod
    def forward(ctx, q_x, mul, add, div, qmin, qmax, signed):
        # PyTorch implementation for training/inference
        input_offset = 0
        output_offset = 0
        rounding = 1
        log2D = int(np.log2(div))
        intermediate = q_x + input_offset * mul + add
        intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) + output_offset
        out = np.clip(intermediate, qmin, qmax)
        return out.astype(q_x.dtype)
    
    @staticmethod
    def symbolic(g, q_x, mul, add, div, qmin, qmax, signed):
        # ONNX export representation
        return g.op("ai.onnx.contrib", q_x, mul, add, div=div, qmin=qmin, qmax=qmax, signed=signed)
    

class Dequant(Function):
    @staticmethod
    def forward(ctx, q_x, scale, zero_point=0.0):
        return (q_x.astype(np.float32) - zero_point) * scale
    
    @staticmethod
    def symbolic(g, q_x, scale, zero_point=0.0):
        # ONNX export representation
        return g.op("ai.onnx.contrib", q_x, scale, zero_point)
    

class Quant(Function):
    @staticmethod
    def forward(ctx, x, scale, zero_point=0.0, signed=True, n_levels=256):
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
    
    @staticmethod
    def symbolic(g, x, scale, zero_point=0.0, signed=True, n_levels=256):
        # ONNX export representation
        return g.op("ai.onnx.contrib", x, scale, zero_point, signed=signed, n_levels=n_levels)
    
    