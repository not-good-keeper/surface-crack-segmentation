"""The six candidate architectures, behind one uniform constructor.

They are chosen to span the axis this benchmark exists to test: **output stride**.
A crack is 1-3 px wide, so any architecture that downsamples to stride 8-32 and
upsamples back has destroyed the evidence before the decoder sees it. Receptive field,
which is what dilated designs optimise for, matters much less than spatial resolution
here.

  U-Net small        stride 1 skips, ~0.5 M params   -- full-control baseline
  U-Net MNv3-Small   stride 1 skips, mobile encoder  -- the obvious edge candidate
  U-Net EffNet-B0    stride 1 skips, strong encoder  -- accuracy ceiling reference
  DeepLabV3+ MNv3    stride 8-16, ASPP context       -- the dilated counter-hypothesis
  LR-ASPP MNv3-Large stride 8-32, very light head    -- genuine mobile seg baseline
  SegFormer-B0       hierarchical transformer, 3.7 M -- attention baseline

If the U-Net family wins clearly, TinySeg should be a U-Net with a slimmed encoder
rather than a DeepLab derivative -- and that is a design decision worth measuring
instead of assuming.
"""
from __future__ import annotations

import torch
import torch.nn as nn


# ------------------------------------------------------------------ tiny U-Net
class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.f(x)


class TinyUNet(nn.Module):
    """Deliberately small, stride-1 skips. This is the shape TinySeg would take."""

    def __init__(self, base=16, depth=4, classes: int = 1):
        super().__init__()
        self.depth = depth
        chs = [base * 2 ** i for i in range(depth + 1)]
        self.enc = nn.ModuleList()
        cin = 3
        for c in chs[:-1]:
            self.enc.append(ConvBlock(cin, c))
            cin = c
        self.bott = ConvBlock(cin, chs[-1])
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            self.up.append(nn.ConvTranspose2d(chs[i + 1], chs[i], 2, stride=2))
            self.dec.append(ConvBlock(chs[i] * 2, chs[i]))
        self.head = nn.Conv2d(chs[0], classes, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        skips = []
        for e in self.enc:
            x = e(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bott(x)
        for u, d, s in zip(self.up, self.dec, reversed(skips)):
            x = u(x)
            x = d(torch.cat([x, s], 1))
        return self.head(x)


# ------------------------------------------------------------------ TinySeg
class SepConv(nn.Module):
    """Depthwise-separable 3x3 + pointwise.

    A plain 3x3 conv at full resolution costs `9*Cin*Cout*H*W` MACs; separating it
    costs `9*Cin*H*W + Cin*Cout*H*W`, roughly 8-9x less at these widths. The
    benchmark showed why this matters: the from-scratch `tiny_unet` has 1.94 M
    parameters yet runs at 96 ms while a 6.25 M EfficientNet-B0 U-Net runs at 69 ms.
    Latency here is dominated by full-resolution 3x3 convolutions, not by parameter
    count, so a "tiny" model has to be tiny in MACs.
    """

    def __init__(self, cin, cout):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(cin, cin, 3, padding=1, groups=cin, bias=False),
            nn.BatchNorm2d(cin), nn.ReLU(inplace=True),
            nn.Conv2d(cin, cout, 1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.f(x)


class TinySeg(nn.Module):
    """U-Net topology, depthwise-separable throughout, narrow at full resolution.

    Shape follows the benchmark result: full-resolution skips generalise across
    materials far better than dilated/coarse-stride heads. Cost follows from keeping
    the widest tensors at the *lowest* resolution -- channels double as resolution
    halves, so the expensive full-res stage runs at `width` channels only.
    """

    def __init__(self, width: int = 16, depth: int = 4, stem: int = 0,
                 classes: int = 1):
        super().__init__()
        stem = stem or width
        chs = [width * 2 ** i for i in range(depth + 1)]
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem, 3, padding=1, bias=False),
            nn.BatchNorm2d(stem), nn.ReLU(inplace=True))
        self.enc = nn.ModuleList()
        cin = stem
        for c in chs[:-1]:
            self.enc.append(SepConv(cin, c))
            cin = c
        self.bott = SepConv(cin, chs[-1])
        self.dec = nn.ModuleList()
        self.red = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            self.red.append(nn.Conv2d(chs[i + 1], chs[i], 1, bias=False))
            self.dec.append(SepConv(chs[i] * 2, chs[i]))
        self.head = nn.Conv2d(chs[0], classes, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.stem(x)
        skips = []
        for e in self.enc:
            x = e(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bott(x)
        for r, d, s in zip(self.red, self.dec, reversed(skips)):
            x = nn.functional.interpolate(r(x), size=s.shape[-2:], mode="nearest")
            x = d(torch.cat([x, s], 1))
        return self.head(x)


# ------------------------------------------------------------------ registry
def build(name: str, classes: int = 1) -> nn.Module:
    """`classes=1` is the binary crack head every stored benchmark used; `classes=3` is
    the deployment head (background / crack / scratch).

    The default stays at 1 deliberately. Every result in data/bench was measured with a
    single-channel sigmoid head, and silently changing what `build(name)` returns would
    make those files describe a model that no longer exists.
    """
    if name.startswith("tinyseg"):
        # tinyseg_w16_d4
        parts = name.split("_")
        w = int([p[1:] for p in parts if p.startswith("w")][0]) if any(
            p.startswith("w") for p in parts) else 16
        d = int([p[1:] for p in parts if p.startswith("d")][0]) if any(
            p.startswith("d") for p in parts) else 4
        return TinySeg(width=w, depth=d, classes=classes)

    if name == "tiny_unet":
        return TinyUNet(base=16, depth=4, classes=classes)

    if name.startswith("smpslim_"):
        # Pretrained encoder + deliberately slim decoder.
        #
        # From-scratch TinySeg reached only 0.06 unseen clDice while a pretrained
        # EfficientNet-B0 U-Net reached 0.29-0.62, so ImageNet features matter more
        # than architecture here. But smp's default U-Net decoder is
        # (256,128,64,32,16) and carries most of the parameters, so the way to stay
        # small without giving up pretraining is to keep the encoder and shrink the
        # decoder -- which also cuts the full-resolution convolutions that dominate
        # CPU latency.
        import segmentation_models_pytorch as smp
        enc = name.split("_", 1)[1]
        return smp.Unet(encoder_name=enc, encoder_weights="imagenet",
                        decoder_channels=(64, 48, 32, 24, 16),
                        in_channels=3, classes=classes)

    if name.startswith("smp_"):
        import segmentation_models_pytorch as smp
        _, arch, enc = name.split("_", 2)
        cls = {"unet": smp.Unet, "dlv3p": smp.DeepLabV3Plus,
               "fpn": smp.FPN}[arch]
        return cls(encoder_name=enc, encoder_weights="imagenet",
                   in_channels=3, classes=classes)

    if name == "lraspp_mnv3":
        from torchvision.models.segmentation import lraspp_mobilenet_v3_large
        m = lraspp_mobilenet_v3_large(weights=None, weights_backbone="DEFAULT",
                                      num_classes=classes)
        return _TVWrap(m)

    if name == "segformer_b0":
        from transformers import SegformerForSemanticSegmentation, SegformerConfig
        cfg = SegformerConfig.from_pretrained("nvidia/mit-b0", num_labels=classes)
        m = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b0", config=cfg, ignore_mismatched_sizes=True)
        return _SegformerWrap(m)

    raise ValueError(f"unknown model {name}")


class _TVWrap(nn.Module):
    """torchvision segmentation models return a dict and a downsampled map."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        out = self.m(x)["out"]
        if out.shape[-2:] != x.shape[-2:]:
            out = nn.functional.interpolate(out, x.shape[-2:], mode="bilinear",
                                            align_corners=False)
        return out


class _SegformerWrap(nn.Module):
    """SegFormer emits logits at stride 4; upsample to full resolution."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        out = self.m(pixel_values=x).logits
        return nn.functional.interpolate(out, x.shape[-2:], mode="bilinear",
                                         align_corners=False)


CANDIDATES = {
    "tiny_unet":          dict(stride="1 (skips)",  note="custom baseline, ~0.5M"),
    "smp_unet_timm-mobilenetv3_small_100":
                          dict(stride="1 (skips)",  note="mobile encoder-decoder"),
    "smp_unet_efficientnet-b0":
                          dict(stride="1 (skips)",  note="accuracy ceiling reference"),
    "smp_dlv3p_timm-mobilenetv3_large_100":
                          dict(stride="8-16",       note="dilated context"),
    "lraspp_mnv3":        dict(stride="8-32",       note="lightest mobile seg head"),
    "segformer_b0":       dict(stride="4",          note="transformer, 3.7M"),
}


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())  # count all: EMA weights have requires_grad=False
