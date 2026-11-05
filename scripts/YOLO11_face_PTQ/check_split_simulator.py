"""Rebuild split INT8 in memory and inspect one image in PC simulation."""
from pathlib import Path
import json
import cv2
import numpy as np
import onnxruntime as ort
from rknn.api import RKNN


def main() -> None:
    """Use the current PTQ defaults without overwriting exported models."""
    import rknn_yolo11_ptq as ptq
    args = ptq.parse_args()
    entries = ptq.validate_calibration_list(args.calibration_list)
    image = cv2.imread(entries[0])
    if image is None:
        raise ValueError(entries[0])
    rgb = cv2.cvtColor(cv2.resize(image, (640, 640)), cv2.COLOR_BGR2RGB)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(args.onnx), sess_options=options,
                                   providers=['CPUExecutionProvider'])
    reference = session.run(None, {session.get_inputs()[0].name:
        rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255})
    runtime = RKNN(verbose=False)
    try:
        assert runtime.config(mean_values=[list(args.mean)],
            std_values=[list(args.std)], target_platform=args.target,
            quantized_dtype='w8a8', quantized_algorithm='normal') == 0
        assert runtime.load_onnx(model=str(args.onnx)) == 0
        assert runtime.build(do_quantization=True,
            dataset=str(args.calibration_list.resolve())) == 0
        assert runtime.init_runtime() == 0
        outputs = runtime.inference(inputs=[rgb[None]], data_format=['nhwc'])
        if outputs is None:
            raise RuntimeError('Simulation failed')
        report = dict(image=entries[0], calibration_images=len(entries),
                      source_onnx=str(args.onnx), backend='PC simulator')
        for label, values in [('onnx', reference), ('simulated_int8', outputs)]:
            report[label] = [dict(shape=list(v.shape), dtype=str(v.dtype),
                minimum=float(v.min()), maximum=float(v.max())) for v in values]
            scores = [v for v in values if tuple(v.shape) == (1, 1, 8400)]
            if len(scores) != 1:
                raise ValueError('Cannot identify scores output')
            score = scores[0]
            report[label + '_scores'] = dict(maximum=float(score.max()),
                nonzero=int(np.count_nonzero(score)),
                above025=int(np.count_nonzero(score >= 0.25)))
        text = json.dumps(report, indent=2)
        Path(__file__).with_name('split_simulator_result.json').write_text(text)
        print(text, flush=True)
    finally:
        runtime.release()


if __name__ == '__main__':
    main()
