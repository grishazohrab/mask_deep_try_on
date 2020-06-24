import dlib
import cv2
from imutils import face_utils

import argparse
import os
import glob

from tqdm import tqdm

detector = dlib.get_frontal_face_detector()


def crop_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 1)

    faces = []
    epsilon = int(gray.shape[0] / 10)

    for (i, rect) in enumerate(rects):
        (x, y, w_, h_) = face_utils.rect_to_bb(rect)

        x = x - epsilon if x - epsilon >= 0 else 0
        y = y - epsilon if y - epsilon >= 0 else 0

        h_ = h_ + epsilon if y + h_ + epsilon < gray.shape[0] else gray.shape[0] - y
        w_ = w_ + epsilon if x + w_ + epsilon < gray.shape[1] else gray.shape[1] - x

        face = img[y:y + h_, x:x + w_]
        faces.append(face)
    return faces


f_parser = argparse.ArgumentParser()

f_parser.add_argument("--with_mask_dir", type=str, default=r"data\with_mask", help="The images with masks")
f_parser.add_argument("--gen_mask_dir", type=str, default=r"data\generated_mask", help="The generated images using cv")
f_parser.add_argument("--without_mask_dir", type=str, default=r"data\without_mask", help="The images without a mask")


if __name__ == '__main__':
    f_args = f_parser.parse_args()

    with_mask_split = os.path.split(f_args.with_mask_dir)
    face_with_mask_dir = os.path.join(with_mask_split[0], "face" + "_" + with_mask_split[1])
    os.makedirs(face_with_mask_dir, exist_ok=True)

    without_mask_split = os.path.split(f_args.without_mask_dir)
    face_without_mask_dir = os.path.join(without_mask_split[0], "face" + "_" + without_mask_split[1])
    os.makedirs(face_without_mask_dir, exist_ok=True)

    gen_mask_split = os.path.split(f_args.gen_mask_dir)
    face_gen_mask_dir = os.path.join(gen_mask_split[0], "face" + "_" + gen_mask_split[1])
    os.makedirs(face_gen_mask_dir, exist_ok=True)

    def crop_faces(imgs_dir, save_dir):
        for i_path in tqdm(glob.glob(os.path.join(imgs_dir, "*.*"))):
            try:
                im_name = os.path.basename(i_path)
                j_path = os.path.join(save_dir, im_name)

                # TODO: only on face in the img
                img = cv2.imread(i_path)
                faces = crop_face(img)
                if len(faces) == 1:
                    cv2.imwrite(j_path, faces[0])
            except Exception as e:
                print(str(e))

    crop_faces(f_args.with_mask_dir, face_with_mask_dir)
    crop_faces(f_args.without_mask_dir, face_without_mask_dir)
    crop_faces(f_args.gen_mask_dir, face_gen_mask_dir)
