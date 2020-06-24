from argparse import ArgumentParser

import torch
from torch import nn

import modules.loss
from models.base_gan import BaseGAN
from modules.wear_mask_modules import MaskTryOnModule

from util.util import un_normalize


class MaskGanModel(BaseGAN):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        parser.add_argument("--face_input",
                            help="how many channels the input should have",
                            default="many",
                            choices=("many", "few"))

        parser.add_argument("--face_w",
                            help="the width of an image",
                            default=128)

        parser.add_argument("--face_h",
                            help="the height of an image",
                            default=256)
        if is_train:
            parser.add_argument(
                "--lambda_ce",
                type=float,
                default=100,
                help="weight for cross entropy loss in final term")

        # based on the num entries in self.visual_names during training
        parser.set_defaults(display_ncols=4)

        # https://stackoverflow.com/questions/26788214/super-and-staticmethod-interaction

        parser = super(MaskGanModel, MaskGanModel).modify_commandline_options(
            parser, is_train
        )

        return parser

    def __init__(self, opt):

        self.channels = 6 if opt.face_input == "many" else 3
        self.face_W = opt.face_w if opt.face_w is None else 128
        self.face_h = opt.face_h if opt.face_h is None else 256

        self.mean = opt.mean
        self.std = opt.std

        BaseGAN.__init__(self, opt)

        self.visual_names = ["with_masks", "images", "gen_images", "fakes"]

        if self.is_train:
            # only show targets during training
            # we use cross entropy loss in both
            self.criterion_CE = nn.CrossEntropyLoss()
            self.loss_names += ["G_ce"]

    def define_G(self):
        """
        The generator is the Warp Module.
        """
        return MaskTryOnModule(channels=self.channels, output_w=self.face_W, output_h=self.face_h)

    def get_D_inchannels(self):
        return 3

    def compute_visuals(self):
        # TODO: how to get mean and std

        self.fakes = un_normalize(self.fakes, self.mean, self.std)
        self.with_masks = un_normalize(self.with_masks, self.mean, self.std)
        self.images = un_normalize(self.images, self.mean, self.std)
        self.gen_images = un_normalize(self.gen_images, self.mean, self.std)

    def set_input(self, input):
        self.with_masks = input["w_img"].to(self.device)
        self.images = input["img"].to(self.device)
        self.gen_images = input["g_img"].to(self.device)

    def forward(self):
        self.fakes = self.net_generator((self.images, self.gen_images))

    def backward_D(self):

        pred_fake = self.net_discriminator(self.fakes.detach())
        self.loss_D_fake = self.criterion_GAN(pred_fake, False)

        pred_real = self.net_discriminator(self.with_masks)
        self.loss_D_real = self.criterion_GAN(pred_real, True)

        self.loss_D = 0.5 * (self.loss_D_fake + self.loss_D_real)

        # calculate gradient penalty
        if any(gp_mode in self.opt.gan_mode for gp_mode in ["gp", "lp"]):
            self.loss_D_gp = (
                modules.loss.gradient_penalty(
                    self.net_discriminator,
                    self.with_masks,
                    self.fakes,
                    self.opt.gan_mode,
                )
                * self.opt.lambda_gp
            )
            self.loss_D += self.loss_D_gp

        # final loss
        self.loss_D.backward()

    def backward_G(self):
        """
        If GAN mode, loss is weighted sum of cross entropy loss and adversarial GAN
        loss. Else, loss is just cross entropy loss.
        """
        # cross entropy loss needed for both gan mode and ce mode
        loss_ce = (
            self.criterion_CE(self.fakes, torch.argmax(self.with_masks, dim=1))   # TODO: what the fuck is this!!!
            * self.opt.lambda_ce
        )

        # if we're in GAN mode, calculate adversarial loss tooy
        self.loss_G_ce = loss_ce  # store loss_ce

        # calculate adversarial loss
        pred_fake = self.net_discriminator(self.fakes)
        self.loss_G_gan = self.criterion_GAN(pred_fake, True) * self.opt.lambda_gan

        # total loss is weighted sum
        self.loss_G = self.loss_G_gan + self.loss_G_ce

        self.loss_G.backward()

    def optimize_parameters(self):
        """
        Optimize both G and D if in GAN mode, else just G.
        Returns:

        """

        super().optimize_parameters()

