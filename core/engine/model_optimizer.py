"""Builds a mixed-precision copy of the inswapper model.

inswapper_128 is dominated by twelve 1024->1024 3x3 convolutions at 32x32, which on Turing and
newer GPUs run ~2.3x faster on FP16 tensor cores. A full FP16 conversion overflows in the
instance-norm style blocks (ReduceMean/Sub/Sqrt/Div) and produces a black image, so only the
Conv nodes are converted and everything else stays FP32. Inputs and outputs stay FP32, so the
model is a drop-in replacement for insightface's INSwapper.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

FP16_OPS = {"Conv"}
# Bump when the conversion changes so stale converted files are rebuilt
MIXED_VERSION = 2


def mixed_precision_path(fp32_path):
    fp32_path = Path(fp32_path)
    return fp32_path.with_name(f"{fp32_path.stem}.mixed-v{MIXED_VERSION}{fp32_path.suffix}")


def _cast_outputs_to_float32(model):
    """keep_io_types leaves an output FP16 when its producer is converted; add a Cast back."""
    from onnx import TensorProto, helper

    for output in model.graph.output:
        tensor_type = output.type.tensor_type
        if tensor_type.elem_type != TensorProto.FLOAT16:
            continue
        fp16_name = output.name + "_fp16"
        for node in model.graph.node:
            node.output[:] = [fp16_name if o == output.name else o for o in node.output]
            node.input[:] = [fp16_name if i == output.name else i for i in node.input]
        model.graph.node.append(helper.make_node("Cast", [fp16_name], [output.name], to=TensorProto.FLOAT))
        tensor_type.elem_type = TensorProto.FLOAT
    return model


def ensure_mixed_precision_model(fp32_path):
    """Return the path of the mixed-precision model, converting it once if needed (None on failure)."""
    fp32_path = Path(fp32_path)
    out_path = mixed_precision_path(fp32_path)
    if out_path.exists() and out_path.stat().st_mtime >= fp32_path.stat().st_mtime:
        return out_path

    try:
        import onnx
        from onnxconverter_common import float16
    except ImportError as e:
        logger.warning("Mixed-precision swapper unavailable (%s); install onnx and onnxconverter-common", e)
        return None

    logger.info("Converting %s to mixed precision (one-time)...", fp32_path.name)
    try:
        model = onnx.load(str(fp32_path))
        block = sorted({node.op_type for node in model.graph.node} - FP16_OPS)
        converted = float16.convert_float_to_float16(model, keep_io_types=True, op_block_list=block)
        converted = _cast_outputs_to_float32(converted)
        tmp_path = out_path.with_suffix(".tmp")
        onnx.save(converted, str(tmp_path))
        tmp_path.replace(out_path)
    except Exception as e:
        logger.warning("Mixed-precision conversion failed, using FP32 model: %s", e)
        return None

    logger.info("Saved %s", out_path.name)
    return out_path
