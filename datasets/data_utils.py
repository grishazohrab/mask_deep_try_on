import os

from torchvision.transforms import ColorJitter, Grayscale, RandomAffine, RandomGrayscale, RandomHorizontalFlip, \
    RandomRotation, RandomVerticalFlip, Normalize, RandomApply, Resize

import random
import numpy as np
import torch


# TODO: add blur
data_transforms = {
    "color_jitter": ColorJitter,
    "random_affine": RandomAffine,
    "random_gray": RandomGrayscale,
    "random_horizontal": RandomHorizontalFlip,
    "random_rotation": RandomRotation,
    "random_vertical": RandomVerticalFlip
}


def generate_rand_numbers(transform_name):
    if transform_name == "color_jitter":
        return {"brightness": random.uniform(0, 1),
                "contrast": random.uniform(0, 1),
                "saturation": random.uniform(0, 1),
                "hue": random.uniform(0, 0.5)}
    elif transform_name == "random_affine":
        return {"shear": random.uniform(0, 90), "degrees": 0}
    elif transform_name == "random_gray":
        return {"p": random.uniform(0, 0.5)}
    elif transform_name == "random_horizontal":
        return {"p": random.uniform(0, 1)}
    elif transform_name == "random_rotation":
        return {"degrees": random.uniform(0, 180)}
    elif transform_name == "random_vertical":
        return {"p": random.uniform(0, 1)}


def get_transforms(transform_names: list, mean: float, std: float, h: int, w: int, rand=True):

    normalize_flag = False
    if "normalize" in transform_names:
        normalize_flag = True
        transform_names.remove("normalize")

    transforms = []
    for t_name in transform_names:
        print(t_name)
        transforms.append(data_transforms[t_name](**generate_rand_numbers(t_name)))

    # rand_apply = RandomApply(transforms, p=0.5)
    rand_apply = None
    if normalize_flag:
        normalize = Normalize(mean, std)
        return rand_apply, Resize((h, w)), normalize
    else:
        return rand_apply, Resize((h, w)), None


def calculate_mean_std_from_pil(images_list):
    means = [[], [], []]
    var_s = [[], [], []]
    for img_ in images_list:
        img = np.array(img_) / 255
        m0 = img[:, :, 0].mean()
        v0 = img[:, :, 0].var()

        m1 = img[:, :, 1].mean()
        v1 = img[:, :, 1].var()

        m2 = img[:, :, 2].mean()
        v2 = img[:, :, 2].var()

        means[0].append(m0)
        var_s[0].append(v0)
        means[1].append(m1)
        var_s[1].append(v1)
        means[2].append(m2)
        var_s[2].append(v2)

    means = np.array(means)
    var_s = np.array(var_s)

    s = var_s.shape[0] / (var_s.shape[0] - 1)

    return {"mean": [means[:, 0].mean(), means[:, 1].mean(), means[:, 2].mean()],
            "std": [np.sqrt(s * var_s[:, 0].mean()), np.sqrt(s * var_s[:, 1].mean()), np.sqrt(s * var_s[:, 2].mean())]}


def remove_extension(fname):
    return os.path.splitext(fname)[0]


