import torch
import torch.nn as nn

# 输入图像大小：32x32
input_tensor = torch.randn(1, 1, 32, 32)  # 1个样本，1个通道，32x32大小

# 卷积核大小：4x4，步幅：1，填充：2
conv_layer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=4, stride=1, padding=2)

# 输出图像大小
output_tensor = conv_layer(input_tensor)

print(f"Input size: {input_tensor.shape}")
print(f"Output size: {output_tensor.shape}")
