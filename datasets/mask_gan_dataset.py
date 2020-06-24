import os
import glob

import numpy as np
from PIL import Image
import cv2
from torchvision.transforms import ToTensor

from datasets import BaseDataset
from datasets.data_utils import calculate_mean_std_from_pil, get_transforms


class MaskGanDataset(BaseDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        # transforms
        parser.add_argument(
            "--input_transforms",
            nargs="+",
            default="none",
            choices=("none", "all"),
            help="what random transforms to perform on the input ('all' for all transforms)",
        )
        if is_train:
            parser.set_defaults(input_transforms=("all"))

        parser.add_argument(
            "--face_w",
            default=128,
            type=int,
            help="the second size of an img"
        )

        parser.add_argument(
            "--face_h",
            default=256,
            type=int,
            help="the first size of an img"
        )

        return parser

    def __init__(self, opt, without_mask_dir=None, generated_mask_dir=None, with_mask_dir=None):
        """

        Args:
            opt: Namespace object
            without_mask_dir (str): optional override path to without_mask_dir
            generated_mask_dir (str): optional override path to generated_mask_dir
            with_mask_dir (str): optional override path to with_mask_dir
        """
        super().__init__(opt)

        self.without_mask_dir = without_mask_dir if without_mask_dir else os.path.join(opt.dataroot, "face_without_mask")
        self.with_mask_dir = with_mask_dir if with_mask_dir else os.path.join(opt.dataroot, "face_with_mask")
        self.generated_mask_dir = generated_mask_dir if generated_mask_dir else \
            os.path.join(opt.dataroot, "face_generated_mask")

        self.face_w = opt.face_w if opt.face_w is not None else 128
        self.face_h = opt.face_h if opt.face_h is not None else 256

        self.samples = []

        self.without_mask_imgs = []
        self.with_mask_imgs = []
        self.gen_mask_imgs = []

        for img_path in glob.glob(os.path.join(self.without_mask_dir, "*.*")):
            img_name = os.path.basename(img_path)

            if img_name in os.listdir(self.generated_mask_dir):
                g_img_path = os.path.join(self.generated_mask_dir, img_name)
                img = Image.open(img_path)
                g_img = Image.open(g_img_path)

                self.without_mask_imgs.append(img)
                self.gen_mask_imgs.append(g_img)

        for img_path in glob.glob(os.path.join(self.with_mask_dir, "*.*")):
            img = Image.open(img_path)

            self.with_mask_imgs.append(img)

        self.mean_std = calculate_mean_std_from_pil(self.with_mask_imgs + self.without_mask_imgs)

        # print(self.mean_std)

        opt.mean = [0, 0, 0]
        opt.std = [1, 1, 1]

        t_names = ["random_vertical"]

        self.transforms = get_transforms(t_names, mean=self.mean_std["mean"], std=self.mean_std["std"], w=self.face_w,
                                         h=self.face_h)

        self.data = []
        for ii in range(len(self.without_mask_imgs)):
            for jj in range(len(self.with_mask_imgs)):
                self.data.append({"img": self.without_mask_imgs[ii],
                                  "g_img": self.gen_mask_imgs[ii],
                                  "w_img": self.with_mask_imgs[jj]})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):

        data_item = self.data[index]
        to_tensor = ToTensor()

        data_item["img"] = to_tensor(self.transforms[0](self.transforms[1](data_item["img"])))
        data_item["g_img"] = to_tensor(self.transforms[0](self.transforms[1](data_item["g_img"])))
        data_item["w_img"] = to_tensor(self.transforms[0](self.transforms[1](data_item["w_img"])))

        if self.transforms[2] is not None:
            data_item["img"] = self.transforms[2](data_item["img"])
            data_item["g_img"] = self.transforms[2](data_item["g_img"])
            data_item["w_img"] = self.transforms[2](data_item["w_img"])

        return data_item
