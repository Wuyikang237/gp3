"""""
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.data import build_dataloader
from ultralytics.utils import DEFAULT_CFG
from fasternet_yolo import FasterNetYOLO

# 实例化模型
model = FasterNetYOLO(nc=2, embed_dim=96, depths=(1,2,8,2))

# 配置训练参数（与 yolo train 命令参数一致）
args = {
    'model': model,                # 传入模型实例
    'data': '/home/wuyikang/finalgp/ultralytics/fasternetdatasets.yaml',  # 数据集配置文件路径
    'epochs': 50,
    'batch': 16,
    'imgsz': 640,
    'device': 0,                   # GPU ID
    'workers': 8,
    'project': 'runs/train',
    'name': 'fasternet_yolo',
    'exist_ok': True,
}

# 创建训练器并开始训练
trainer = DetectionTrainer(overrides=args)
trainer.train()
"""""
"""""
from fasternet_yolo import FasterNetYOLO

# 实例化模型
model = FasterNetYOLO(nc=2, embed_dim=96, depths=(1,2,8,2))

# 直接调用 train 方法
model.train(
    data='/home/wuyikang/finalgp/ultralytics/fasternetdatasets.yaml',
    epochs=50,
    batch=16,
    imgsz=640,
    device='cuda'  # 或 'cpu'
)
"""
"""""
import torch
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.utils import DEFAULT_CFG
from fasternet_yolo import FasterNetYOLO  # 确保该文件在根目录

# 1. 实例化自定义模型
model = FasterNetYOLO(nc=2, embed_dim=96, depths=(1,2,8,2))

# 2. 设置训练参数
args = {
    'data': '/home/wuyikang/finalgp/ultralytics/fasternetdatasets.yaml',   # 你的数据集配置
    'epochs': 50,
    'batch': 16,
    'imgsz': 640,
    'device': 0,                          # GPU ID，如果使用CPU则设为'cpu'
    'workers': 8,
    'project': 'runs/train',
    'name': 'fasternet_yolo',
    'exist_ok': True,
}

# 3. 创建训练器并设置模型
trainer = DetectionTrainer(overrides=args)
trainer.model = model.to(trainer.device)   # 将模型移到设备上
trainer.train()
"""
import torch
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.modules.fasternet_simAM_yolo import FasterNetYOLO  # 你的自定义模型

# 1. 实例化模型
model = FasterNetYOLO(
    nc=2,                # 类别数（根据你的数据集修改）
    embed_dim=96,
    depths=(1, 2, 8, 2),
    mlp_ratio=2.0
)

# 2. 训练参数（与 YOLO 命令行参数一致）
args = {
    'model': 'yolov8n.pt',               # 提供一个有效路径，避免初始化错误
    'data': '/home/wuyikang/finalgp/ultralytics/fasternetdatasets.yaml',  # 你的数据集配置文件
    'epochs': 50,
    'batch': 16,
    'imgsz': 640,
    'device': 0,                         # GPU ID，若用CPU则设为 'cpu'
    'workers': 8,
    'project': 'runs/train',
    'name': 'fasternet_yolo',
    'exist_ok': True,
}

# 3. 创建训练器
trainer = DetectionTrainer(overrides=args)

# 4. 替换模型为你自己的模型（放在训练器设备上）
trainer.model = model.to(trainer.device)

# 5. 开始训练（训练器会重新初始化优化器、损失等，基于新模型）
trainer.train()