"""Usine ViTMatte-S (Apache 2.0) -> ONNX, taille fixe = demi-résolution portrait du rush (544x960, multiple de 32)."""
import torch, warnings, sys
warnings.filterwarnings('ignore')
from transformers import VitMatteForImageMatting
m = VitMatteForImageMatting.from_pretrained('hustvl/vitmatte-small-composition-1k').eval()
class W(torch.nn.Module):
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, x): return s.m(pixel_values=x).alphas
H, Wd = int(sys.argv[1]), int(sys.argv[2])
x = torch.rand(1, 4, H, Wd)  # RGB normalisé + trimap
with torch.no_grad():
    torch.onnx.export(W(m), x, f'vitmatte_s_{H}x{Wd}.onnx', input_names=['input'], output_names=['alpha'], opset_version=17, dynamo=False)
print('ok')
