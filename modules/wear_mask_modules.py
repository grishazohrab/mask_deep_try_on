import logging

from torch import nn
import torch

from modules.layers import UNetDown, UNetUp
from modules.pix2pix_modules import define_G

wm_log = logging.getLogger("mask_try_on_module_shape")


class MaskTryOnModule(nn.Module):

    def __init__(self, output_w, output_h, dropout=0.5, channels=3):
        super(MaskTryOnModule, self).__init__()

        self.face_output = define_G(input_nc=channels, output_nc=3, ngf=64, netG="unet_128")

    def forward(self, face):
        face = torch.cat([face[0], face[1]], dim=1)

        face_o = self.face_output(face)
        return face_o
