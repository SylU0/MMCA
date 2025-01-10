import json
import os
from os.path import join

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.loss import LabelSmoothingCrossEntropy

from models.model_utils import print_rank_0, main_process, vis_masked_image


class MaskCrossAttention(nn.Module):
    def __init__(self, Lq, Lkv, Cq, Ckv, embed_dim, num_heads,
                 attn_drop_rate=0.2, proj_drop_rate=0, pos=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.q = nn.Linear(Cq, embed_dim)
        self.kv = nn.Linear(Ckv, 2 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.scale = head_dim ** -0.5

        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj_drop = nn.Dropout(proj_drop_rate)
        self.softmax = nn.Softmax(dim=-1)
        self.pos = pos
        if pos:
            self.attn_pos = nn.Parameter(torch.zeros((1, self.num_heads, Lq, Lkv)))

    def softmax_with_mask(self, attn, mask):
        """mask.shape: [B, Lq, 1]"""
        _, nh, _, Lkv = attn.shape

        # ---- Step1: 构造对 attn 原有部分的掩码 (mask) ----
        mask = mask.unsqueeze(1).expand(-1, nh, -1, -1)  # [B, nh, Lq, 1]
        # ①1 -> 0；②0 -> -100
        attn_mask = (1 - mask).expand(-1, -1, -1, Lkv) * -100.0
        attn += attn_mask

        # ---- Step2: dummy 列赋值 ----
        # 若 mask的值=0: dummy_col = 100 (希望它吸收掉该行几乎全部注意力)
        # 若 mask的值=1: dummy_col = -100 (dummy 列几乎被忽略)
        dummy_col = torch.where(mask == 0, 100.0, -100.0)

        # ---- Step3: 拼接 extra 列 ----
        attn = torch.cat([attn, dummy_col], dim=-1)  # [B, nh, Lq, Lkv+1]

        # ---- Step4: Softmax ----
        attn = self.softmax(attn)

        # ---- Step5: 去除 dummy 列，恢复原始大小 [B, nh, Lq, Lkv] ----
        attn = attn[..., :Lkv]

        return attn

    def forward(self, Xq, Xkv, mask=None):
        B, Lq, _ = Xq.shape
        B, Lkv, _ = Xkv.shape
        num_heads = self.num_heads

        # [B, Lq, d] -> [B, Lq, nh, hd] -> [B, nh, Lq, hd]
        Q = self.q(Xq).reshape(B, Lq, num_heads, -1).permute(0, 2, 1, 3)
        # [B, Lkv, 2d] -> [B, Lkv, 2, nh, hd] -> [2, B, nh, Lkv, hd]
        KV = self.kv(Xkv).reshape(B, Lkv, 2, num_heads, -1).permute(2, 0, 3, 1, 4)
        K, V = KV[0], KV[1]

        Q = Q * self.scale
        attn = Q @ K.transpose(-2, -1)  # [B, nh, Lq, Lkv]
        if self.pos:
            attn = attn + self.attn_pos

        if mask is not None:
            attn = self.softmax_with_mask(attn, mask)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        out = attn @ V  # [B, nh, Lq, hd]
        out = out.transpose(1, 2).reshape(B, Lq, -1)  # [B, Lq, d]
        out = self.proj(out)
        out = self.proj_drop(out)

        return out


class MaskFeatureAggregate(nn.Module):
    def __init__(self, dim, L, p, pos=True):
        super().__init__()
        self.pos = pos
        self.fc = nn.Linear(dim, p)
        if pos:
            self.attn_pos = nn.Parameter(torch.zeros((1, L, p)))
        self.softmax = nn.Softmax(dim=-1)
        self.act = nn.GELU()

    def forward(self, x, mask=None):
        """mask: BL1"""
        # BLC -> BLp
        attn = self.act(self.fc(x))
        if self.pos:
            attn = attn + self.attn_pos
        # BLp -> BpL
        attn = attn.permute(0, 2, 1)

        if mask is not None:
            attn = attn.masked_fill(mask.permute(0, 2, 1) == 0, -100)

        attn = self.softmax(attn)
        # BpL @ BLC -> BpC
        return attn @ x


class DWSep2DConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DWSep2DConv, self).__init__()
        self.dw_conv = nn.Conv2d(
            in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
            stride=stride, padding=padding,
            groups=in_channels  # 每个输入通道单独卷积
        )
        self.pw_conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=1  # 1x1 卷积
        )

    def forward(self, x):
        x = self.dw_conv(x)  # 深度卷积
        x = self.pw_conv(x)  # 逐点卷积
        return x


class LocalFeatureAggregate(nn.Module):
    def __init__(self, dim, p, kernel_size, depth_wise=False):
        super().__init__()
        padding = (kernel_size - 1) // 2
        if depth_wise:
            self.conv = DWSep2DConv(in_channels=dim, out_channels=p, kernel_size=kernel_size,
                                    stride=1, padding=padding)
        else:
            self.conv = nn.Conv2d(in_channels=dim, out_channels=p, kernel_size=kernel_size,
                                  stride=1, padding=padding)
        self.softmax = nn.Softmax(dim=-1)
        self.act = nn.GELU()

    def forward(self, x):
        B, L, C = x.shape
        H = W = int(L ** 0.5)

        # 计算注意力图, 形状：[B, p, H, W]
        attn = self.conv(x.transpose(1, 2).view(B, C, H, W))
        attn = self.act(attn)
        # 展平空间维度, 形状：[B, p, L]
        attn = attn.view(B, -1, L)
        attn = self.softmax(attn)
        return attn @ x


class PredictorLG(nn.Module):
    """ Importance Score Predictor"""

    def __init__(self, embed_dim=384):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU()
        )
        self.out_conv = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 2),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, input_x, mask=None):
        """
            input_x: BLC
            mask: BL1
        """
        x = self.in_conv(input_x)
        B, L, C = x.size()
        local_x = x[:, :, :C // 2]

        if mask is not None:
            global_x = (x[:, :, C // 2:] * mask).sum(
                dim=1, keepdim=True) / torch.sum(mask, dim=1, keepdim=True)
        else:
            global_x = torch.mean(x[:, :, C // 2:], keepdim=True, dim=1)

        x = torch.cat([local_x, global_x.expand(B, L, C // 2)], dim=2)
        pred_score = self.out_conv(x)

        mask = F.gumbel_softmax(pred_score, hard=True, dim=2)[:, :, 0:1]
        return mask


class MyModel(nn.Module):
    def __init__(self, backbone, dim, num_classes, last_size, num_heads,
                 keep_ratio_li, loss_lam_li, loss_keep_lam_li, loss_sim_lam_li, p_li,
                 attn_drop_rate=0.2, proj_drop_rate=0, head_drop=0, agg_pos=True,
                 cross_attn_pos=True, sim_loss_type="mse", label_smooth=0, training=False
                 ):
        super(MyModel, self).__init__()
        print_rank_0("==================== MyModel_v1 ====================")
        print_rank_0(f"Key parameters:\n"
                     f"p_li: {p_li}, agg_pos: {agg_pos}, cross_attn_pos: {cross_attn_pos}\n"
                     f"loss_lam_li: {loss_lam_li}\n"
                     f"keep_ratio_li: {keep_ratio_li}, loss_keep_lam_li: {loss_keep_lam_li}\n"
                     f"sim_loss_type: {sim_loss_type}, loss_sim_lam_li: {loss_sim_lam_li}\n")
        if training:
            from setup import config
            my_config = {
                "p_li": p_li, "agg_pos": agg_pos, "cross_attn_pos": cross_attn_pos,
                "loss_lam_li": loss_lam_li,
                "keep_ratio_li": keep_ratio_li, "loss_keep_lam_li": loss_keep_lam_li,
                "sim_loss_type": sim_loss_type, "loss_sim_lam_li": loss_sim_lam_li,
            }
            # 保存到 JSON 文件
            os.makedirs(config.data.log_path, exist_ok=True)
            output_path = join(config.data.log_path, "MyModel_config.json")
            with open(output_path, "w", encoding="utf-8") as json_file:
                json.dump(my_config, json_file, indent=4, ensure_ascii=False)

        self.backbone = backbone
        self.loss_lam_li = loss_lam_li
        self.keep_ratio_li = keep_ratio_li
        self.loss_keep_lam_li = loss_keep_lam_li
        self.sim_loss_type = sim_loss_type.lower()
        self.loss_sim_lam_li = loss_sim_lam_li

        cross_layer_ratio = [8, 4, 2, 1]
        dim_li = [dim // cross_layer_ratio[i] for i in range(4)]
        size_li = [last_size * cross_layer_ratio[i] for i in range(4)]
        len_li = [size_li[i] ** 2 for i in range(4)]
        dim_sum = sum(dim_li)
        self.size_li = size_li
        self.cross_layer_ratio = cross_layer_ratio

        # norm
        self.norm_li = nn.ModuleList([
            nn.LayerNorm(dim_li[i]) for i in range(3)
        ])
        # 掩码预测模块(mask predictor module)
        self.mask_predictor_li = nn.ModuleList([
            PredictorLG(embed_dim=dim_li[i]) for i in range(4)
        ])
        # 掩码特征聚集模块(mask feature aggregate module)
        self.mask_feature_aggregate_li = nn.ModuleList([
            MaskFeatureAggregate(dim=dim_li[i], L=len_li[i], p=p_li[i], pos=agg_pos)
            for i in range(4)
        ])
        # 掩码交叉注意力机制模块(mask cross attention module)
        self.mask_cross_attn_li = nn.ModuleList([
            MaskCrossAttention(Lq=len_li[-1], Lkv=p_li[i], Cq=dim, Ckv=dim_li[i],
                               embed_dim=dim_li[i], num_heads=num_heads, pos=cross_attn_pos,
                               attn_drop_rate=attn_drop_rate, proj_drop_rate=proj_drop_rate)
            for i in range(4)
        ])

        """4、辅助分类器损失"""
        if loss_lam_li[3] != 0:
            self.head_4 = nn.Linear(dim, num_classes)

        self.act = nn.GELU()
        self.norm = nn.LayerNorm(dim_sum)
        self.head_drop = nn.Dropout(head_drop)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(dim_sum, num_classes)

        self.ce_loss_fn = LabelSmoothingCrossEntropy(smoothing=label_smooth)

    def forward(self, x, label=None, access=False, epoch=0):
        B = x.size(0)
        loss_lam_li = self.loss_lam_li
        size_li = self.size_li
        x_li = self.backbone(x)

        # norm
        for i in range(3):
            x_li[i] = self.norm_li[i](x_li[i])

        # 预测4个stage的掩码mask
        mask_li = []
        for i in range(4):
            mask_li.append(self.mask_predictor_li[i](x_li[i]))

        if access and main_process():
            from setup import config
            for i in range(4):
                save_dir = join(config.data.log_path, "masked_images", f"stage_{i + 1}")
                vis_masked_image(
                    x[:6].clone().detach(),
                    mask_li[i][:6].clone().detach(),
                    save_dir=save_dir, epoch=epoch)

        # 提前拿到query
        x_q = x_li[-1]
        mask_4 = mask_li[-1]
        # 将4个stage的特征进行聚集
        for i in range(4):
            x_li[i] = self.mask_feature_aggregate_li[i](x_li[i], mask=mask_li[i])

        # 最后一个stage做Q的交叉注意力机制
        for i in range(4):
            x_li[i] = self.mask_cross_attn_li[i](Xq=x_q, Xkv=x_li[i], mask=mask_4)

        # 拼接结果
        out = torch.cat(x_li, dim=-1)  # [B, L, sum(dim_li)]

        out = self.act(out)
        out = self.norm(out)
        out = self.head_drop(out)
        # 全局平均池化
        out = torch.flatten(self.avg_pool(out.transpose(-2, -1)), 1)
        out = self.head(out)  # [B, num_classes]

        if self.training and label is not None:
            loss_li = []
            """1、分类损失"""
            loss_ce = self.ce_loss_fn(out, label)
            loss_ce = loss_lam_li[0] * loss_ce
            loss_li.append(loss_ce)

            """2、mask保留损失"""
            if loss_lam_li[1] != 0:
                loss_ratio = 0.0
                for i in range(4):
                    loss_ratio += self.loss_keep_lam_li[i] * (
                            (mask_li[i].mean(dim=1) - self.keep_ratio_li[i]) ** 2).mean()
                loss_ratio = loss_lam_li[1] * loss_ratio
                loss_li.append(loss_ratio)

            """3、相似度损失"""
            if loss_lam_li[2] != 0:
                pivot_mask_li = []
                # B,L4,1 -> B,L4 -> B,1,L4 -> B,H4,W4
                last_mask = mask_li[-1].squeeze(-1).view(B, size_li[-1], size_li[-1])
                for i in range(3):
                    ratio = self.cross_layer_ratio[i]
                    # B, H4, W4 -> B, Hi, Wi
                    pivot_mask_li.append(last_mask.repeat_interleave(
                        ratio, dim=1).repeat_interleave(ratio, dim=2))

                loss_sim = 0.0
                for i in range(3):
                    if self.sim_loss_type == "mse":
                        # B,Li,1 -> B,Li -> B,1,Li -> B,Hi,Wi
                        mask_li[i] = mask_li[i].squeeze(-1).view(B, size_li[i], size_li[i])
                        loss_sim += self.loss_sim_lam_li[i] * torch.mean(
                            (mask_li[i] - pivot_mask_li[i]) ** 2)
                    elif self.sim_loss_type == "cosine":
                        # B,Li,1 -> B,Li
                        mask_li[i] = mask_li[i].squeeze(-1)
                        # B,Hi,Wi -> B,Li
                        pivot_mask_li[i] = pivot_mask_li[i].view(B, -1)
                        loss_sim += (1 - self.loss_sim_lam_li[i] * F.cosine_similarity(
                            pivot_mask_li[i], mask_li[i], dim=1).mean())
                    elif self.sim_loss_type == "bce":
                        # B,Li -> B,Hi,Wi
                        mask_li[i] = mask_li[i].squeeze(-1).view(B, size_li[i], size_li[i])
                        loss_sim += self.loss_sim_lam_li[i] * F.binary_cross_entropy_with_logits(
                            pivot_mask_li[i], mask_li[i])
                    else:
                        raise ValueError("Unsupported similarity loss type")
                loss_sim = loss_lam_li[2] * loss_sim
                loss_li.append(loss_sim)

            """4、辅助分类器损失"""
            if loss_lam_li[3] != 0:
                # 最后一个stage的特征做辅助分类
                x_q = mask_4 * x_q
                out_4 = torch.flatten(self.avg_pool(x_q.transpose(-2, -1)), 1)
                out_4 = self.head_4(out_4)
                loss_aux = self.ce_loss_fn(out_4, label)
                loss_aux = loss_lam_li[3] * loss_aux
                loss_li.append(loss_aux)

            loss_sum = sum(loss_li)
            loss_li.insert(0, loss_sum)
            return out, loss_li
        else:
            return out


def test():
    from mytimm.models import create_model
    from models.backbone.Swin_Transformer import swin_backbone

    backbone = create_model('swin_base_patch4_window12_384.ms_in22k',
                            pretrained=False,
                            num_classes=200,
                            drop_path_rate=0.2, cross_layer=True)

    backbone = swin_backbone(num_classes=200,
                             drop_path_rate=0.2,
                             img_size=384,
                             window_size=12)

    model = MyModel(backbone=backbone, dim=1024,
                    last_size=12, num_heads=32, num_classes=200,
                    agg_pos=True, cross_attn_pos=True, training=False,
                    p_li=[144, 144, 144, 144],
                    loss_lam_li=[1, 2, 1, 0],
                    keep_ratio_li=[0.5, 0.5, 0.5, 0.5],
                    loss_keep_lam_li=[1, 1, 1, 1],
                    sim_loss_type="bce",
                    loss_sim_lam_li=[1, 1, 1]
                    )

    x = torch.randn(2, 3, 384, 384)
    label = torch.randint(200, (2,))

    model = model.to("cuda")
    x = x.to("cuda")
    label = label.to("cuda")

    y = model(x, label)
    print(y[0].shape)
    print(y[1])


if __name__ == "__main__":
    test()
