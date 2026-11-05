"""Split decoded YOLO face boxes and scores into separate ONNX outputs."""

import argparse
from pathlib import Path

import onnx
from onnx import helper

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "models/yolov11n/yolo_face/yolo11n_face.onnx"


def split_model(source: Path, destination: Path) -> None:
    """Expose the final Concat inputs without changing model weights."""
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite: {destination}")
    model = onnx.shape_inference.infer_shapes(onnx.load(str(source)))
    if len(model.graph.output) != 1:
        raise ValueError("Expected a single-output source model")
    output = model.graph.output[0]
    producers = [n for n in model.graph.node if output.name in n.output]
    if len(producers) != 1:
        raise ValueError("Cannot identify final output producer")
    concat = producers[0]
    axis = next((a.i for a in concat.attribute if a.name == "axis"), None)
    if concat.op_type != "Concat" or axis != 1 or len(concat.input) != 2:
        raise ValueError("Expected final Concat(axis=1) with two inputs")
    info = {v.name: v for v in model.graph.value_info}
    tensors = [info[name] for name in concat.input]
    shapes = [
        [d.dim_value for d in v.type.tensor_type.shape.dim] for v in tensors
    ]
    if shapes != [[1, 4, 8400], [1, 1, 8400]]:
        raise ValueError(f"Unexpected boxes/scores shapes: {shapes}")
    if any(output.name in n.input for n in model.graph.node):
        raise ValueError("Final Concat output has other consumers")
    occupied = {name for n in model.graph.node for name in n.output}
    if occupied.intersection({"boxes", "scores"}):
        raise ValueError("Output names boxes/scores already exist")
    model.graph.node.remove(concat)
    del model.graph.output[:]
    # Identity nodes give the two outputs clear public names.
    for name, value, shape in zip(("boxes", "scores"), tensors, shapes):
        model.graph.node.append(helper.make_node(
            "Identity", [value.name], [name], name=f"export_{name}"
        ))
        model.graph.output.append(helper.make_tensor_value_info(
            name, value.type.tensor_type.elem_type, shape
        ))
    onnx.checker.check_model(model)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(destination))
    print(f"Saved: {destination}")
    print("boxes: (1, 4, 8400); scores: (1, 1, 8400)")


def main() -> None:
    """Parse paths and generate a new two-output face detector."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    destination = args.output or args.input.with_name(
        args.input.stem + "_split.onnx"
    )
    split_model(args.input, destination)


if __name__ == "__main__":
    main()
