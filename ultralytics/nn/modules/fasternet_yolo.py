import torch
import torch.nn as nn
from ultralytics.nn.modules import Detect, C2f, Conv
from ultralytics.utils.torch_utils import initialize_weights
from fasternet import FasterNet

class FasterNetYOLO(nn.Module):
    def __init__(self, nc=80, ch=3, embed_dim=96, depths=(1,2,8,2), mlp_ratio=2., **kwargs):
        super().__init__()
        self.nc = nc
        self.ch = ch
        self.end2end = False
        self.args = {}
        self.names = {i: f"{i}" for i in range(nc)}

        # 主干网络
        self.backbone = FasterNet(
            in_chans=ch,
            embed_dim=embed_dim,
            depths=depths,
            mlp_ratio=mlp_ratio,
            fork_feat=True,
            **kwargs
        )

        # 计算各阶段输出通道
        c2 = embed_dim * 2   # 8倍下采样
        c3 = embed_dim * 4   # 16倍下采样
        c4 = embed_dim * 8   # 32倍下采样

        # 检测头
        self.head = self._build_head(c2, c3, c4, nc)

        # 模拟官方模型结构，使损失函数能正确访问检测头
        self.model = [self.head['detect']]
        self.register_buffer('stride', torch.tensor([32.]))

        # 延迟创建损失函数，避免设备不一致
        self.criterion = None

        initialize_weights(self)

    def _build_head(self, c2, c3, c4, nc):
        conv_c2 = Conv(c2, 256, 1)
        conv_c3 = Conv(c3, 512, 1)
        conv_c4 = Conv(c4, 1024, 1)
        upsample = nn.Upsample(scale_factor=2, mode='nearest')
        c2f_1 = C2f(1024 + 512, 512)   # 输入1536 -> 512
        c2f_2 = C2f(512 + 256, 256)    # 输入768 -> 256
        c2f_3 = C2f(256 + 512, 512)    # 输入768 -> 512
        c2f_4 = C2f(512 + 1024, 1024)  # 输入1536 -> 1024
        down_1 = Conv(256, 256, 3, 2)
        down_2 = Conv(512, 512, 3, 2)
        detect = Detect(nc=nc, ch=(256, 512, 1024))
        return nn.ModuleDict({
            'conv_c2': conv_c2, 'conv_c3': conv_c3, 'conv_c4': conv_c4,
            'upsample': upsample,
            'c2f_1': c2f_1, 'c2f_2': c2f_2, 'c2f_3': c2f_3, 'c2f_4': c2f_4,
            'down_1': down_1, 'down_2': down_2,
            'detect': detect,
        })

    def forward(self, x, **kwargs):
        if isinstance(x, dict):          # 训练模式
            if self.criterion is None:
                from ultralytics.utils.loss import v8DetectionLoss
                self.criterion = v8DetectionLoss(self)
            preds = self._forward_once(x['img'])
            return self.criterion(preds, x)   # 返回损失
        else:                            # 推理/验证模式
            return self._forward_once(x)

    def loss(self, batch, preds=None):
        """验证时计算损失"""
        if preds is None:
            preds = self._forward_once(batch['img'])
        if self.criterion is None:
            from ultralytics.utils.loss import v8DetectionLoss
            self.criterion = v8DetectionLoss(self)
        loss_total, loss_items = self.criterion(preds, batch)  # 直接返回两个值
        return loss_total, loss_items

    def _forward_once(self, x):
        features = self.backbone.forward_det(x)   # [f0, f1, f2, f3]
        f1, f2, f3 = features[1], features[2], features[3]  # P3, P4, P5

        p3 = self.head['conv_c2'](f1)   # 256
        p4 = self.head['conv_c3'](f2)   # 512
        p5 = self.head['conv_c4'](f3)   # 1024

        up4 = self.head['upsample'](p5)
        cat4 = torch.cat([up4, p4], dim=1)      # 1536
        x4 = self.head['c2f_1'](cat4)           # 512

        up3 = self.head['upsample'](x4)
        cat3 = torch.cat([up3, p3], dim=1)      # 768
        x3 = self.head['c2f_2'](cat3)           # 256

        down3 = self.head['down_1'](x3)
        cat_down4 = torch.cat([down3, x4], dim=1)  # 768
        x4_new = self.head['c2f_3'](cat_down4)     # 512

        down4 = self.head['down_2'](x4_new)
        cat_down5 = torch.cat([down4, p5], dim=1)  # 1536
        x5 = self.head['c2f_4'](cat_down5)         # 1024

        out = self.head['detect']([x3, x4_new, x5])
        return out