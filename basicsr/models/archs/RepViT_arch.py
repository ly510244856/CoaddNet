# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# 加入LayerNorm重参数设计的代码，对应NAFunireplkln_net.py
# ------------------------------------------------------------------------

'''
Simple Baselines for Image Restoration

@article{chen2022simple,
  title={Simple Baselines for Image Restoration},
  author={Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},
  journal={arXiv preprint arXiv:2204.04676},
  year={2022}
}
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
import sys
sys.path.append(".")
from basicsr.models.archs.arch_util import LayerNorm2d
from basicsr.models.archs.local_arch import Local_Base

# mine codes
from basicsr.models.archs.RepViTnet import NAFBlock, SCABlock, TCNAFBlock
from basicsr.models.archs.dysample import DySample

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

# ============================
# downsample
# ============================
class Conv2d_LN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('ln', LayerNorm2d(b))
        torch.nn.init.constant_(self.ln.weight, bn_weight_init)
        torch.nn.init.constant_(self.ln.bias, 0)

class Residual(torch.nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)

class Downsample(nn.Module):
    def __init__(self, inp, oup, kernel_size, use_se, use_hs):
        super(Downsample, self).__init__()
        self.token_mixer = nn.Sequential(
            Conv2d_LN(inp, inp, kernel_size, 2, (kernel_size - 1) // 2, groups=inp),
            SCABlock(inp) if use_se else nn.Identity(),
            Conv2d_LN(inp, oup, ks=1, stride=1, pad=0)
        )
        self.channel_mixer = Residual(nn.Sequential(
                # pw
                Conv2d_LN(oup, 2 * oup, 1, 1, 0),
                nn.GELU() if use_hs else SimpleGate(),
                # pw-linear
                Conv2d_LN(oup, oup, 1, 1, 0, bn_weight_init=0),
            ))
        
    def forward(self, x):
        return self.channel_mixer(self.token_mixer(x))

class NAFRepViT(nn.Module):

    def __init__(self, inp_shape=(1,256,256), width=16, block_type="NAF", down_type="conv",
                 middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[], dilated=True, deploy=False, kernel_size=3,
                 up_type="pixelshuffle"
                ):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=inp_shape[0], out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=inp_shape[0], kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan, dilated=dilated, kernel_size=kernel_size) if block_type=="NAF" else TCNAFBlock(chan, dilated=dilated,deploy=deploy, kernel_size=kernel_size) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2) if down_type=="conv" else \
                    Downsample(chan, 2*chan, 3, use_se=True, use_hs=False)
            )
            chan = chan * 2
        
        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan, dilated=dilated, kernel_size=kernel_size) if block_type=="NAF" else TCNAFBlock(chan, dilated=dilated,deploy=deploy, kernel_size=kernel_size) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            if up_type=="pixelshuffle":
                self.ups.append(
                    nn.Sequential(
                        nn.Conv2d(chan, chan * 2, 1, bias=False),
                        nn.PixelShuffle(2)

                        # nn.ConvTranspose2d(chan, chan//2, kernel_size=2, stride=2, padding=0)
                    )
                )
            elif up_type=="dysample":
                self.ups.append(
                    nn.Sequential(
                        nn.Conv2d(chan, chan // 2, 1, bias=False),
                        DySample(chan//2, style='pl', dyscope=True)
                    )
                )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan, dilated=dilated, kernel_size=kernel_size) if block_type=="NAF" else TCNAFBlock(chan, dilated=dilated,deploy=deploy, kernel_size=kernel_size) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

        self.apply(self._init_weights)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x
    
    def reparameterize_unireplknet(self):
        for m in self.modules():
            if hasattr(m, 'reparameterize'):
                m.reparameterize()

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)

class NAFNetLocal(Local_Base, NAFRepViT):
    def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        NAFRepViT.__init__(self, *args, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)


if __name__ == '__main__':
    inp_shape=(4,256,256)
    width = 32

    enc_blks = [2, 2, 4, 8]
    middle_blk_num = 12
    dec_blks = [2, 2, 2, 2]

    block_type = "NC"
    down_type = "conv"
    dilated = False
    deploy = True
    up_type="dysample"
    
    net = NAFRepViT(inp_shape=inp_shape, width=width, middle_blk_num=middle_blk_num,
                       block_type=block_type,down_type=down_type,
                       enc_blk_nums=enc_blks, dec_blk_nums=dec_blks,
                       dilated=dilated,deploy=deploy,
                       up_type=up_type)


    inp_shape = (4, 256, 256)

    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)

    params = float(params[:-3])
    macs = float(macs[:-4])

    print(macs, params)
    print(net)

    for m in net.modules():
        if hasattr(m, 'reparameterize_unireplknet'):
            m.reparameterize_unireplknet()

    macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)
    params = float(params[:-3])
    macs = float(macs[:-4])
    print(macs, params)
    # print(net)
