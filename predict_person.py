import _locale
_locale._getdefaultlocale = (lambda *args: ['zh_CN', 'utf8'])
import  os
import  re
import  sys
from ultralytics import YOLO
import yaml
import argparse
from pathlib import Path

from PIL import Image
import numpy as np
from pathlib import Path
from typing import Union
import cv2



def test_video():
    model = YOLO(r"C:\Users\14685\Desktop\jianzhibest.pt")
    #C:\Users\14685\Desktop\yolov8\prune.pt
    # 测试视频存放目录Q
    #pa = "rtsp://op:aotto355@192.168.200.50:554/Streaming/Channels/101" # menkou
    pa = "rtsp://op:aotto355@192.168.8.100:554/Streaming/Channels/101" #changqu
    #pa = "rtsp://op:aotto355@192.168.8.100:554/Streaming/Channels/401" #tuyou
    cap = cv2.VideoCapture(pa)
    # 调用设备自身摄像头
    # cap = cv2.VideoCapture(0) # -1
    # 设置视频尺寸
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),)
    # 第一个参数是将检测视频存储的路径
    out = cv2.VideoWriter('save.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20, size)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            res = model(frame,conf = 0.3)
            ann = res[0].plot()
            cv2.imshow("yolo", ann)
            out.write(ann)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()
    cap.release()


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

parser = argparse.ArgumentParser()

parser.add_argument('--task', type=str, default='detect', help='task type')

parser.add_argument('--hyp', type=str, default=ROOT / 'shiliyu/hyp.scratch.yaml', help='hyperparameters path')

parser.add_argument('--model', nargs='+', type=str, default=r"D:\yolo11\model\best_renyuan.pt", help='model path(s)')

parser.add_argument('--source', type=str, default=r'C:\Users\sly\Desktop\2025-03-25\rechengxing\images', help='file/dir/URL/glob/screen/0(webcam)')

parser.add_argument('--imgsz', type=int, default=640, help='size of each image dimension')

parser.add_argument('--conf', type=float, default=0.3, help='confidence threshold')
parser.add_argument('--iou', type=float, default=0.1, help='NMS IoU threshold')
parser.add_argument('--device', type=str, default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--project', default=ROOT / 'run', help='save results to project/name')
parser.add_argument('--name', type=str, default='predict', help='exp name')
parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
parser.add_argument('--save', default=True,  help='baocuntupian')
#parser.add_argument('--show', default=False,  help='show results if possible')

parser.add_argument('--save_txt',default=True, action='store_true', help='save results to *.txt')

#parser.add_argument('--opset',default='11', action='store_true', help='opset vision')
parser.add_argument('--classes',default=[0,1,2], action='store_true', help='类别')
#parser.add_argument('--visualize',default=False, action='store_true', help='特征可视化')
args = parser.parse_args()

assert args.source, 'argument --source path is required'
assert args.model, 'argument --model path is required'

if __name__ == '__main__':
    # Initialize
    args.model = str(args.model)
    model = YOLO(args.model)
    hyperparams = yaml.safe_load(open(args.hyp))

    hyperparams['task'] = args.task

    hyperparams['source'] = args.source
    hyperparams['imgsz'] = args.imgsz
    hyperparams['conf'] = args.conf
    hyperparams['iou'] = args.iou
    hyperparams['device'] = args.device
    hyperparams['project'] = args.project
    hyperparams['name'] = args.name
    hyperparams['resume'] = args.resume
    hyperparams['save'] = args.save
    hyperparams['save_txt'] = args.save_txt
    #hyperparams['show'] = args.show
    hyperparams['classes'] = args.classes
    #hyperparams['visualize'] = args.visualize

    model.predict(**hyperparams)
    #test_video()

