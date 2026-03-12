import numpy as np
from onnxruntime_extensions import onnx_op, PyOp, get_library_path
import onnxruntime as ort
import onnx
from onnx import helper, TensorProto

@onnx_op(op_type="Quant", domain="ai.onnx.contrib", inputs=[PyOp.dt_float, PyOp.dt_float, PyOp.dt_int64, PyOp.dt_int64], outputs=[PyOp.dt_float])
def Quant(x, scale, n_levels, signed, **kwargs):
    print("IN FUNCTION:", n_levels, signed, kwargs)
    return x

# When building the node:
n_levels_tensor = helper.make_tensor("n_levels", TensorProto.INT64, [1], [999])
signed_tensor = helper.make_tensor("signed", TensorProto.INT64, [1], [1])

graph = helper.make_graph(
    [
        helper.make_node(
            "Quant", ["input", "scale", "n_levels", "signed"], ["output"],
            domain="ai.onnx.contrib"
        )
    ],
    "test",
    [
        helper.make_tensor_value_info("input", TensorProto.FLOAT, [None]),
        helper.make_tensor_value_info("scale", TensorProto.FLOAT, [None]),
        helper.make_tensor_value_info("n_levels", TensorProto.INT64, [1]),
        helper.make_tensor_value_info("signed", TensorProto.INT64, [1]),
    ],
    [helper.make_tensor_value_info("output", TensorProto.FLOAT, [None])],
    initializer=[n_levels_tensor, signed_tensor]
)
model = helper.make_model(graph, opset_imports=[
    helper.make_opsetid("", 18),
    helper.make_opsetid("ai.onnx.contrib", 1)
])

so = ort.SessionOptions()
so.register_custom_ops_library(get_library_path())
sess = ort.InferenceSession(model.SerializeToString(), so)
x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
s = np.array([1.0, 1.0, 1.0], dtype=np.float32)
sess.run(None, {"input": x, "scale": s})