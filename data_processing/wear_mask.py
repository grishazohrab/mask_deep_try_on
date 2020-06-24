from imutils import face_utils
from sympy import Line
import numpy as np
import dlib
import cv2

import argparse
import glob
import os
import tqdm


detector = dlib.get_frontal_face_detector()


def get_face_landmarks(image, shape_predictor=r"data_processing/shape_predictor_68_face_landmarks.dat"):

    predictor = dlib.shape_predictor(shape_predictor)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)

    faces_shape = []

    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        faces_shape.append(shape)

    return faces_shape


def get_rotate_angle(shape):
    s_p_1 = shape[31]
    s_p_2 = shape[9]
    line_1 = Line(s_p_1, s_p_2)
    line_2 = Line((0, 0), (1, 0))
    return line_1.angle_between(line_2)


def get_affine_coordinate(shape):
    src_tri = np.array([[0, 0], [1, 0], [0, 1]]).astype(np.float32)
    dst_tri = np.array([shape[31], shape[16], shape[9]]).astype(np.float32)

    warp_mat = cv2.getAffineTransform(src_tri, dst_tri)
    return warp_mat


def mask_try_on(img_path, mask_img, save_path):
    image = cv2.imread(img_path)
    mask = cv2.imread(mask_img)
    shapes = get_face_landmarks(image)
    width = int(shapes[0][15][0] - shapes[0][1][0])
    height = int(shapes[0][8][1] - shapes[0][30][1])
    dim = (width, height)
    mask = cv2.resize(mask, dim)

    # warp_mat = get_affine_coordinate(shapes[0])
    # warp_dst = cv2.warpAffine(mask, warp_mat, (mask.shape[1], mask.shape[0]))
    warp_dst = mask
    # cv2.imshow('Warp mask', warp_dst)

    center = (warp_dst.shape[1] // 2, warp_dst.shape[0] // 2)
    angle = float(get_rotate_angle(shapes[0]))
    scale = 1

    rot_mat = cv2.getRotationMatrix2D(center, angle, scale)
    warp_rotate_dst = cv2.warpAffine(warp_dst, rot_mat, (warp_dst.shape[1], warp_dst.shape[0]))
    img_part = image[shapes[0][1][1]:shapes[0][1][1] + height, shapes[0][1][0]:shapes[0][1][0] + width]

    for ii in range(0, warp_rotate_dst.shape[0]):
        for jj in range(0, warp_rotate_dst.shape[1]):
            if not (((0, 0, 0) < warp_rotate_dst[ii][jj]).all() and (warp_rotate_dst[ii][jj] < (255, 255, 255)).all()):
                warp_rotate_dst[ii][jj] = img_part[ii][jj]

    image[shapes[0][1][1]:shapes[0][1][1] + height, shapes[0][1][0]:shapes[0][1][0] + width] = warp_rotate_dst

    cv2.imwrite(save_path, image)


wear_mask_parser = argparse.ArgumentParser()
wear_mask_parser.add_argument("--images_dir", type=str, default=r"data\without_mask", help="Images without mask")
wear_mask_parser.add_argument("--mask_dir", type=str, default=r"data\generated_mask", help="Images with mask")
wear_mask_parser.add_argument("--mask", type=str, default=r"data_processing\masks\mask_1.png", help="Path of the mask")


if __name__ == '__main__':
    args = wear_mask_parser.parse_args()
    os.makedirs(args.mask_dir, exist_ok=True)

    for image_p in tqdm.tqdm(glob.glob(args.images_dir + "/*.*")):
        img_name = os.path.basename(image_p)
        s_img_p = os.path.join(args.mask_dir, img_name)
        try:
            mask_try_on(image_p, args.mask, s_img_p)
        except Exception as e:
            print(str(e))
