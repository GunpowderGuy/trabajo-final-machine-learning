import torch
import torch.nn as nn
from torchvision.ops import deform_conv2d

# Definimos un wrapper para deformable conv
class DeformConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size * kernel_size, kernel_size=3, padding=1)
        self.padding = padding

    def forward(self, x):
        offset = self.offset_conv(x)
        return deform_conv2d(x, offset, self.weight, self.bias, padding=self.padding)

# Definimos un modelo de prueba simple
class MiniDeformNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.defconv = DeformConv2D(3, 8)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(8, 2)

    def forward(self, x):
        x = self.relu(self.defconv(x))
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)

# Probar el modelo con datos aleatorios
model = MiniDeformNet()
x = torch.randn(4, 3, 32, 32)  # batch de 4 imágenes RGB 32x32
output = model(x)

print("✅ Deformable Conv funciona correctamente.")
print("Output shape:", output.shape)
