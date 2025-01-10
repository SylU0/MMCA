import os
from os.path import join
from torchvision import transforms
import torch


def main_process():
    return ("LOCAL_RANK" not in os.environ) or (
            "LOCAL_RANK" in os.environ and int(os.environ["LOCAL_RANK"]) == 0)


def print_rank_0(*args, **kwargs):
    if main_process():
        print(*args, **kwargs)


def denormalize(tensor, mean=None, std=None):
    # mean, std 是RGB三个通道的列表
    # tensor: [3, H, W]
    if std is None:
        std = [0.229, 0.224, 0.225]
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    # 将范围限制在[0,1]
    tensor.clamp_(0, 1)
    return tensor


@torch.no_grad()
def vis_masked_image(x, mask, save_dir, epoch):
    B, _, img_size, _ = x.shape
    L = mask.shape[1]
    H = W = int(L ** 0.5)

    patch_size = img_size // H  # 每个patch的边长像素数

    # 将mask变为[B,H,W]，去掉最后的1维
    mask = mask.squeeze(-1).reshape(B, H, W)  # [B,H,W]
    # 上采样 mask
    mask = mask.repeat_interleave(patch_size, dim=1)  # 在 H 方向重复
    mask = mask.repeat_interleave(patch_size, dim=2)  # 在 W 方向重复
    # 最终是 [B, img_size, img_size]
    mask = mask.unsqueeze(1)  # [B, 1, img_size, img_size]

    # 将输入图像与掩码相乘，获得被mask掉的图像
    masked_image = x * mask

    # 将图像移至CPU并detach
    masked_image = masked_image.cpu()

    to_pil = transforms.ToPILImage()
    # 逐张逆归一化并保存
    for i in range(B):
        img = masked_image[i]  # [3, H, W]
        # 逆归一化
        img = denormalize(img)
        # 转换为PIL格式
        img = to_pil(img)
        # 保存
        save_path = join(save_dir, f"epoch_{epoch}")
        os.makedirs(save_path, exist_ok=True)
        img.save(join(save_path, f"{i}.png"))


def save_sample_map(sample_map):
    from setup import config
    map_file_path = join(config.data.log_path, "visualize", "sampling_map", "map_file")
    os.makedirs(map_file_path, exist_ok=True)
    if sample_map.shape[-1] == 9216:
        torch.save(sample_map, join(map_file_path, "part_sample_1.pt"))
    elif sample_map.shape[-1] == 2304:
        torch.save(sample_map, join(map_file_path, "part_sample_2.pt"))
    elif sample_map.shape[-1] == 576:
        torch.save(sample_map, join(map_file_path, "part_sample_3.pt"))
    elif sample_map.shape[-1] == 144:
        torch.save(sample_map, join(map_file_path, "part_sample_4.pt"))
