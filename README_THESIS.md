# YOLOv8-FasterNet-SimAM 毕设成果说明

## 1. 课题简介
本项目面向绝缘子及缺陷检测任务，在 YOLOv8 框架上进行结构改进与轻量化优化，核心目标是在保证检测性能的同时降低模型部署成本。

## 2. 主要改进点
1. 引入 FasterNet 骨干网络
2. 集成 SimAM 注意力机制
3. 进行模型剪枝（20%/30%/40%）并微调
4. 进行知识蒸馏（两组 KD 参数）

## 3. 代码入口
- 改进模型实现：`ultralytics/nn/modules/fasternet_simAM_yolo.py`
- 剪枝脚本：`tools/prune_finetune.py`
- 蒸馏脚本：`tools/distill_train.py`
- 结果总表：`../results_comparison.md`
- 实验输出：`runs/healthcheck/`

## 4. 实验设置
- 数据集：500 train / 100 val
- 类别：insulator、defect
- 输入尺寸：640x640
- 训练设备：CUDA GPU

## 5. 结果结论（摘要）
- 剪枝方案表现突出，验证了模型冗余可被有效压缩。
- 蒸馏方案在当前学生模型配置下仍有提升空间。
- 当前最佳部署候选：剪枝+微调后的模型（以结果总表为准）。

## 6. 推荐答辩展示顺序
1. 任务背景与痛点
2. 方法总览（FasterNet + SimAM + 剪枝 + 蒸馏）
3. 消融实验设计
4. 对比结果与可视化
5. 结论与后续优化方向

## 7. 单独成果仓库建议（可选）
建议新建独立仓库仅展示你的工作内容，目录可为：
```text
thesis-yolo-fasternet-simam/
  README.md
  code/
    fasternet_simAM_yolo.py
    prune_finetune.py
    distill_train.py
  experiments/
    commands.md
    configs/
  results/
    comparison_table.md
    curves/
  docs/
    slides_outline.md
```

## 8. 致谢说明（建议）
本项目基于 Ultralytics YOLOv8 开源框架开展二次开发与实验验证。
