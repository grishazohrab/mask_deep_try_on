import logging

from torch import nn
import torch

from modules.layers import UNetDown, UNetUp

wm_log = logging.getLogger("mask_try_on_module_shape")


class MaskTryOnModule(nn.Module):

    def __init__(self, output_w, output_h, dropout=0.5, channels=3):
        super(MaskTryOnModule, self).__init__()

        self.face_down1 = UNetDown(channels, 64, normalize=False)
        self.face_down2 = UNetDown(64, 128)
        self.face_down3 = UNetDown(128, 256)
        self.face_down4 = UNetDown(256, 512)
        self.face_down5 = UNetDown(512, 1024, dropout=dropout)
        self.face_down6 = UNetDown(1024, 1024, normalize=False, dropout=dropout)
        # the two UNetUp's below will be used WITHOUT concatenation.
        # hence the input size will not double
        self.face_up1 = UNetUp(1024, 1024)
        self.face_up2 = UNetUp(1024, 512)
        self.face_up3 = UNetUp(512, 256)
        self.face_up4 = UNetUp(256, 128)
        self.face_up5 = UNetUp(128, 64)

        self.upsample_and_pad = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(3 * 64, 3, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, face):

        face = torch.cat([face[0], face[1]], dim=1)

        wm_log.debug("face shape:", face.shape)

        wm_log.debug("shapes should match except in the channel dim")

        face_d1 = self.face_down1(face)

        face_d2 = self.face_down2(face_d1)
        wm_log.debug("face_d2 shape, should be 128 channel:", face_d2.shape)

        face_d3 = self.face_down3(face_d2)
        wm_log.debug("face_d3 shape, should be 256 channel:", face_d3.shape)

        face_d4 = self.face_down4(face_d3)
        wm_log.debug("face_d4 shape, should be 512 channel:", face_d4.shape)

        face_d5 = self.face_down5(face_d4)
        wm_log.debug("face_d5 shape, should be 1024 channel:", face_d5.shape)

        face_d6 = self.face_down6(face_d5)
        wm_log.debug("face_d6 shape, should be 1024 channel:", face_d6.shape)

        face_u1 = self.face_up1(face_d6, None)
        wm_log.debug("face_u1 shape, should be 1024 channel:", face_u1.shape)

        face_u2 = self.face_up2(face_u1, face_d4)
        wm_log.debug("face_u2 shape, should be 512 channel:", face_u2.shape)

        face_u3 = self.face_up3(face_u2, face_d3)
        wm_log.debug("face_u3 shape, should be 256 channel:", face_u3.shape)

        face_u4 = self.face_up4(face_u3, face_d2)
        wm_log.debug("face_u4 shape, should be 128 channel:", face_u4.shape)

        face_u5 = self.face_up4(face_u4, face_d1)
        wm_log.debug("face_u5 shape, should be 64 channel:", face_u5.shape)

        upsampled = self.upsample_and_pad(face_u5)

        wm_log.debug("upsampled shape, should be original channel:", upsampled.shape)
        return upsampled
