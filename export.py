import _locale
_locale._getdefaultlocale = (lambda *args: ['zh_CN', 'utf8'])

import sys
import os
from pathlib import Path
from ultralytics import YOLO
import yaml
import argparse
# v8trans
import onnx
import onnx.helper as helper


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def v8trans(file):

    prefix, suffix = os.path.splitext(file)
    dst = prefix + ".transd" + suffix

    model = onnx.load(file)
    node  = model.graph.node[-1]

    old_output = node.output[0]
    node.output[0] = "pre_transpose"

    for specout in model.graph.output:
        if specout.name == old_output:
            shape0 = specout.type.tensor_type.shape.dim[0]
            shape1 = specout.type.tensor_type.shape.dim[1]
            shape2 = specout.type.tensor_type.shape.dim[2]
            new_out = helper.make_tensor_value_info(
                specout.name,
                specout.type.tensor_type.elem_type,
                [0, 0, 0]
            )
            new_out.type.tensor_type.shape.dim[0].CopyFrom(shape0)
            new_out.type.tensor_type.shape.dim[2].CopyFrom(shape1)
            new_out.type.tensor_type.shape.dim[1].CopyFrom(shape2)
            specout.CopyFrom(new_out)

    model.graph.node.append(
        helper.make_node("Transpose", ["pre_transpose"], [old_output], perm=[0, 2, 1])
    )

    print(f"Model save to {dst}")
    onnx.save(model, dst)
    return 0

parser = argparse.ArgumentParser()

parser.add_argument('--hyp', type=str, default=ROOT / 'shiliyu/hyp.scratch.yaml', help='hyperparameters path')

parser.add_argument('--format', type=str, default='onnx', help='export model')

parser.add_argument('--model', nargs='+', type=str, default=r'C:\Users\sly\Desktop\liaopian\exp2\weights\best.pt', help='model path(s)')
parser.add_argument('--onnxmodel', nargs='+', type=str, default=r'C:\Users\sly\Desktop\liaopian\exp2\weights\best.onnx', help='model path(s)')

parser.add_argument('--imgsz', type=int, default=256, help='size of each image dimension')
#parser.add_argument('--nms', type=str, default=False, help='is nms')

parser.add_argument('--device', type=str, default='0', help='device')

parser.add_argument('--opset', type=str, default=13, help='onnx banben')

parser.add_argument('--dynamic', type=str, default=False, help='is dynamic model')

parser.add_argument('--worksapce', type=str, default=1, help='xianchengshu')

parser.add_argument('--simplify', type=str, default=True, help='')

parser.add_argument('--batch', type=int, default=1, help='')
args = parser.parse_args()

assert args.model, 'argument --model path is required'

if __name__ == '__main__':
    #Initialize
    args.model = str(args.model)
    model = YOLO(args.model)
    hyperparams = yaml.safe_load(open(args.hyp))
    hyperparams['format'] = args.format
    hyperparams['imgsz'] = args.imgsz
    # hyperparams['nms'] = args.nms
    hyperparams['dynamic'] = args.dynamic
    hyperparams['simplify'] = args.simplify
    hyperparams['device'] = args.device
    hyperparams['opset'] = args.opset
    hyperparams['batch'] = args.batch
    model.export(**hyperparams)


    # model123 = onnx.load(r'C:\Users\sly\Desktop\xiaozi\exp\weights\best.onnx')
    # model123.ir_version = 7
    # onnx.save_model(model123, r'C:\Users\sly\Desktop\xiaozi\exp\weights\best2.onnx')

    # 调整onnx模型输出的通道顺序，，例如 由[1,38,34000]转为[1,34000,38]
    # 单通道模型需修改 exporter中的通道数
    # 转换后的模型为transd.onnx
    #v8trans(args.onnxmodel)

# import torch
# from ultralytics import YOLO

# model = YOLO(r"D:\MODEL\MODEL_sly\renyuan_640_p5_10.28\exp\weights\best.pt")
# dummy_input = torch.randn(4, 3, 640, 640)  # Batch size of 4, RGB 640x640 input
# model.export(format="onnx", opset=11, input=dummy_input)