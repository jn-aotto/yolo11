import _locale
_locale._getdefaultlocale = (lambda *args: ['zh_CN', 'utf8'])
from ultralytics import YOLO
import yaml
import argparse
from pathlib import Path
import sys
import os

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

parser = argparse.ArgumentParser()

parser.add_argument('--task', type=str, default='segment', help='task type')

parser.add_argument('--model', type=str, default=ROOT / '../../train_model/yolov8m-seg.pt', help='initial weights path')
parser.add_argument('--cfg', type=str, default=ROOT / 'shiliyu/yolov8m-seg.yaml', help='model.yaml path')
parser.add_argument('--data', type=str, default=ROOT / 'shiliyu/zijide_kai_hua_seg.yaml', help='dataset.yaml path')
parser.add_argument('--hyp', type=str, default=ROOT / 'shiliyu/hyp.scratch.yaml', help='hyperparameters path')

parser.add_argument('--epochs', type=int, default=100, help='total training epochs')
parser.add_argument('--batch_size', type=int, default=8, help='total batch size for all GPUs, -1 for autobatch')
parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=1280, help='train, val image size (pixels)')
parser.add_argument('--rect', action='store_true', help='rectangular training')
parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
parser.add_argument('--noval', action='store_true', help='only validate final epoch')
parser.add_argument('--noautoanchor', action='store_true', help='disable AutoAnchor')
parser.add_argument('--noplots', action='store_true', help='save no plot files')
parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')
parser.add_argument('--image-weights', default=False, action='store_true', help='use weighted image selection for training')
parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='optimizer')
parser.add_argument('--sync_bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')

parser.add_argument('--workers', type=int, default=4, help='max dataloader workers (per RANK in DDP mode)')

parser.add_argument('--project', default=r'D:\MODEL\MODEL_sly\kailiehuahen_1280_seg_p345_09.30', help='save to project/name')

parser.add_argument('--name', default='exp', help='save to project/name')
parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
parser.add_argument('--quad', action='store_true', help='quad dataloader')
parser.add_argument('--cos-lr', action='store_true', help='cosine LR scheduler')
parser.add_argument('--label-smoothing', type=float, default=0.05, help='Label smoothing epsilon')
parser.add_argument('--patience', type=int, default=0, help='EarlyStopping patience (epochs without improvement)')
parser.add_argument('--freeze', nargs='+', type=int, default=[0,1,2,3,4,5], help='Freeze layers: backbone=10, first3=0 1 2')
parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
parser.add_argument('--seed', type=int, default=0, help='Global training seed')
parser.add_argument('--local_rank', type=int, default=-1, help='Automatic DDP Multi-GPU argument, do not modify')

parser.add_argument('--amp', action='store_true', default=False, help='Unuse Automatic Mixed Precision (AMP) training')
args = parser.parse_args()

assert args.data, 'argument --data path is required'
assert args.model, 'argument --model path is required'

if __name__ == '__main__':
    # Initialize
    #先在trainer中稀疏训练

    #engine model不从yaml加载模型
    #util loss定制device

    args.model = str(args.model)
    args.cfg = str(args.cfg)
    model = YOLO(args.cfg).load(args.model) # 从网络模型配置文件中加载模型

#------------# 直接加载模型
    #model = YOLO(args.model) 
#-------------

    #model = YOLO(r"D:\MODEL\MODEL_sly\kailiehuahen_1280_seg_p345_09.06\exp\weights\last.pt") # 直接加载模型
    

    hyperparams = yaml.safe_load(open(args.hyp))

    hyperparams['task'] = args.task
    hyperparams['data'] = args.data

    hyperparams['epochs'] = args.epochs
    hyperparams['batch'] = args.batch_size
    hyperparams['imgsz'] = args.imgsz
    hyperparams['device'] = args.device
    hyperparams['project'] = args.project
    hyperparams['name'] = args.name
    hyperparams['resume'] = args.resume
    hyperparams['patience'] = args.patience
    hyperparams['workers'] = args.workers
    hyperparams['amp'] = args.amp
    hyperparams['freeze'] = args.freeze

    model.train(**hyperparams)