from ultralytics import YOLO
from fasternet_yolo import FasterNetYOLO  # 导入自定义模型

# 实例化模型
model = FasterNetYOLO(nc=2, embed_dim=96, depths=(1,2,8,2))

# 或者通过 YOLO 类加载（需要先保存为 .pt 文件）
# 直接使用 YOLO 类包装：
yolo_model = YOLO(model)  # 此时 model 是 nn.Module

# 训练
yolo_model.train(data='/home/wuyikang/finalgp/ultralytics/fasternetdatasets.yaml', epochs=50)