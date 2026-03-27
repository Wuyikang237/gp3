"""
from ultralytics.data import YOLODataset

# 替换为你的 YAML 文件路径
yaml_path = '/home/wuyikang/finalgp/ultralytics/fasternetdatasets.yaml'

dataset = YOLODataset(data=yaml_path, task='detect')
print(f"训练集样本数: {len(dataset)}")
"""
from ultralytics import YOLO

# 使用一个简单的模型（例如 yolov8n.pt）来验证数据集
model = YOLO('yolov8n.pt')
results = model.val(data='/home/wuyikang/finalgp/ultralytics/fasternetdatasets.yaml')
print("数据集验证完成")