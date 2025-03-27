# YOLO11 项目

本项目基于 YOLO11 框架，包含模型训练、预测和导出功能，支持分类、分割等任务。

## 项目结构
yolo11/ 
├── export.py # 模型导出脚本（支持 ONNX 格式） 
├── export_tensorrt.py # 模型导出脚本（支持 TensorRT 格式） 
├── predict_cls.py # 分类任务预测脚本 
├── predict_kailiehuahen.py # 开裂划痕检测任务预测脚本 
├── predict_person.py # 人员检测任务预测脚本 
├── train.py # 通用训练脚本 
├── train-cls.py # 分类任务训练脚本 
├── train_kailiehuahen.py # 开裂划痕检测任务训练脚本 
├── model/ # 模型文件存储目录 
│ ├── yolo11m-seg.pt # 分割模型 
│ ├── yolo11x-cls.pt # 分类模型 
│ └── ... # 其他模型文件 
├── shiliyu/ # 示例配置文件目录 
│ ├── hyp.scratch.yaml # 超参数配置文件 
│ ├── yolo11m-seg.yaml # 分割模型配置文件 
│ ├── yolo11x-cls.yaml # 分类模型配置文件 
│ └── ... # 其他配置文件 
└── ultralytics/ # YOLO 框架相关代码
## 环境依赖

- Python 3.8+
- PyTorch
- OpenCV
- ultralytics

安装依赖：
```bash
pip install -r requirements.txt
使用说明
1. 模型训练
通用训练
运行 train.py 脚本：python [train.py](http://_vscodecontentref_/9) --data <数据集配置文件> --model <模型路径>
分类任务训练
运行 train-cls.py 脚本：python  --data <数据集配置文件> --model <模型路径>
开裂划痕检测任务训练
运行 train_kailiehuahen.py 脚本：python  --data <数据集配置文件> --model <模型路径>


2. 模型预测
分类任务预测
运行 predict_cls.py 脚本：
python  --source <输入数据路径> --model <模型路径>
开裂划痕检测任务预测
运行 predict_kailiehuahen.py 脚本：python  --source <输入数据路径> --model <模型路径>

人员检测任务预测
运行 predict_person.py 脚本：python [predict_person.py](http://_vscodecontentref_/14) --source <输入数据路径> --model <模型路径>

3. 模型导出
导出为 ONNX 格式
运行 export.py 脚本：python [export.py](http://_vscodecontentref_/15) --model <模型路径> --format onnx

导出为 TensorRT 格式
运行 export_tensorrt.py 脚本：python [export_tensorrt.py](http://_vscodecontentref_/16) --model <模型路径> --format engine

注意事项
请根据实际需求修改脚本中的默认参数。
确保数据集路径和模型路径正确无误。
贡献
欢迎提交问题和贡献代码！

许可证
本项目遵循 MIT 许可证。 ```