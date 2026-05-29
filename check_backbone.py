import torch
from ultralytics.nn.backbone.efficientViT import EfficientViT_M0
from ultralytics.nn.backbone.fasternet import fasternet_t0
from ultralytics.nn.backbone.mobilenetv4 import MobileNetV4ConvSmall
from ultralytics.nn.backbone.repvit import repvit_m0_9
from ultralytics.nn.backbone.rmt import RMT_T
from ultralytics.nn.backbone.SwinTransformer import SwinTransformer_Tiny

x = torch.randn(1, 3, 640, 640)

print("=== EfficientViT_M0 ===")
model = EfficientViT_M0()
outs = model(x)
for i, o in enumerate(outs):
    print(f"Layer {i}: {o.shape}")

print("\n=== fasternet_t0 ===")
model = fasternet_t0()
outs = model(x)
for i, o in enumerate(outs):
    print(f"Layer {i}: {o.shape}")

print("\n=== MobileNetV4ConvSmall ===")
model = MobileNetV4ConvSmall()
outs = model(x)
for i, o in enumerate(outs):
    print(f"Layer {i}: {o.shape}")

print("\n=== repvit_m0_9 ===")
model = repvit_m0_9()
outs = model(x)
for i, o in enumerate(outs):
    print(f"Layer {i}: {o.shape}")

print("\n=== RMT_T ===")
model = RMT_T()
outs = model(x)
for i, o in enumerate(outs):
    print(f"Layer {i}: {o.shape}")

print("\n=== SwinTransformer_Tiny ===")
model = SwinTransformer_Tiny()
outs = model(x)
for i, o in enumerate(outs):
    print(f"Layer {i}: {o.shape}")