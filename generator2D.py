import glob
import json
import os
import pathlib
import sys
from typing import Iterable

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from rotatable_image import RotatableImage


def rand_between(lower, upper):
    return lower + np.random.rand() * (upper - lower)


def perturb_hsv(image_bgr, perturbation_bounds):
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.int32)
    image_hsv += np.round([rand_between(*bounds) for bounds in perturbation_bounds]).astype(np.int32)
    image_hsv = np.clip(image_hsv, a_min=0, a_max=255).astype(np.uint8)

    image_bgr_perturbed = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)

    return image_bgr_perturbed


def load_rotatable_images(images_dir: str):
    rotatable_images = {}
    for filepath in glob.glob(os.path.join(images_dir, "*")):
        try:
            image = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
            rotatable_image = RotatableImage(image)
        except:
            continue

        label = pathlib.Path(filepath).stem
        rotatable_images[label] = rotatable_image

    return rotatable_images


def generate_background(
    min_random_width: int,
    max_random_width: int,
    min_random_height: int,
    max_random_height: int,
    output_width: int,
    output_height: int,
):
    base_width = int(np.round(rand_between(min_random_width, max_random_width)))
    base_height = int(np.round(rand_between(min_random_height, max_random_height)))
    base_texture = np.round(np.random.random([base_height, base_width, 3]) * 255).astype(np.uint8)

    base_texture_resized = cv2.resize(base_texture, dsize=[output_height, output_width], interpolation=cv2.INTER_LINEAR)

    return base_texture_resized


def rotate_image(rotatable_image: RotatableImage, euler_bounds_xyz, origin_uv_bounds):
    euler_angles = [rand_between(*bounds) for bounds in euler_bounds_xyz]
    rotation_matrix = Rotation.from_euler("xyz", euler_angles).as_matrix()

    origin_uv = [rand_between(*bounds) for bounds in origin_uv_bounds]
    rotated_image = rotatable_image.get_rotated_image(rotation_matrix, origin_uv)

    return rotated_image


class LayeredImage:
    def __init__(self, background: np.ndarray):
        self.canvas_, self.layer_properties_ = background, []

    @property
    def composite(self):
        return self.canvas_

    @property
    def layer_properties(self):
        return self.layer_properties_

    def add_layer(self, image_id, image_bgra: np.ndarray, visibility_threshold: float, attempt_limit: int):
        image, image_mask = image_bgra[..., :3], image_bgra[..., 3] != 0
        image_center_x_offset_limit = int(np.round(image.shape[1] * (visibility_threshold - 0.5)))
        image_center_y_offset_limit = int(np.round(image.shape[0] * (visibility_threshold - 0.5)))

        attempt_count = 0
        while attempt_count < attempt_limit:
            canvas_slice, image_slice = self._get_random_image_slice(
                image_center_x_offset_limit, image_center_y_offset_limit, image.shape
            )

            canvas_mask = np.zeros(self.canvas_.shape[:2], bool)
            canvas_mask[canvas_slice] = image_mask[image_slice]

            other_masks_visible_new = []
            for layer_properties in self.layer_properties_:
                other_visible_mask_new = layer_properties["mask_visible"].copy()
                other_visible_mask_new[canvas_mask] = False
                if np.count_nonzero(other_visible_mask_new) >= layer_properties["mask_visible_threshold"]:
                    other_masks_visible_new.append(other_visible_mask_new)
                else:
                    break
            if len(other_masks_visible_new) < len(self.layer_properties_):
                attempt_count += 1
                continue

            for layer_properties, mask_visible_new in zip(self.layer_properties_, other_masks_visible_new):
                layer_properties["mask_visible"] = mask_visible_new
            self.layer_properties_.append(
                {
                    "id": image_id,
                    "mask_whole": canvas_mask,
                    "mask_visible": canvas_mask,
                    "mask_visible_threshold": int(np.count_nonzero(canvas_mask[canvas_slice] * visibility_threshold)),
                }
            )

            self.canvas_[canvas_slice][image_mask[image_slice]] = image[image_slice][image_mask[image_slice]]
            return True

        return False

    def _get_random_image_slice(
        self,
        image_center_x_offset_limit: int,
        image_center_y_offset_limit: int,
        image_shape: Iterable[int],
    ):
        image_center_x_offset = int(
            np.round(rand_between(image_center_x_offset_limit, self.canvas_.shape[1] - image_center_x_offset_limit))
        )
        image_center_y_offset = int(
            np.round(rand_between(image_center_y_offset_limit, self.canvas_.shape[0] - image_center_y_offset_limit))
        )
        half_image_shape = np.ceil(np.array(image_shape[:2], float) / 2).astype(np.int32)

        canvas_xmin_raw = image_center_x_offset - half_image_shape[1]
        canvas_ymin_raw = image_center_y_offset - half_image_shape[0]
        canvas_xmin = max(0, canvas_xmin_raw)
        canvas_ymin = max(0, canvas_ymin_raw)
        canvas_xmax = min(self.canvas_.shape[1], canvas_xmin + image_shape[1] - (canvas_xmin - canvas_xmin_raw))
        canvas_ymax = min(self.canvas_.shape[0], canvas_ymin + image_shape[0] - (canvas_ymin - canvas_ymin_raw))
        image_xmin = half_image_shape[1] - (image_center_x_offset - canvas_xmin)
        image_ymin = half_image_shape[0] - (image_center_y_offset - canvas_ymin)
        image_xmax = image_xmin + (canvas_xmax - canvas_xmin)
        image_ymax = image_ymin + (canvas_ymax - canvas_ymin)

        canvas_slice = np.index_exp[canvas_ymin:canvas_ymax, canvas_xmin:canvas_xmax]
        image_slice = np.index_exp[image_ymin:image_ymax, image_xmin:image_xmax]

        return canvas_slice, image_slice


class Generator2D:
    def __init__(self, config_filepath: str):
        self.config_ = json.load(open(config_filepath, "r"))
        self.objects_ = load_rotatable_images(self.config_["io"]["template_dir"])

    def generate(self):
        if not os.path.exists(self.config_["io"]["output_dir"]):
            os.mkdir(self.config_["io"]["output_dir"])

        labels = sorted(self.objects_.keys())

        layered_image = LayeredImage(self._generate_background())
        for i in range(3):
            layered_image.add_layer(
                image_id=0,
                image_bgra=self._randomize_rotatable_image(self.objects_[labels[0]]),
                visibility_threshold=self.config_["composition"]["layering"]["visibility_threshold"],
                attempt_limit=self.config_["composition"]["layering"]["attempt_limit"],
            )
        cv2.imwrite("test.png", layered_image.composite)

    def _perturb_hsv(self, image_bgr):
        return perturb_hsv(image_bgr, self.config_["hsv_perturbation_bounds"])

    def _generate_background(self):
        return generate_background(
            **self.config_["background"],
            output_width=self.config_["io"]["width"],
            output_height=self.config_["io"]["height"]
        )

    def _randomize_rotatable_image(self, rotatable_image: RotatableImage):
        image = rotate_image(rotatable_image, **self.config_["objects"]["rotation_randomization"])

        image[..., :3] = self._perturb_hsv(image[..., :3])

        fx = rand_between(*self.config_["objects"]["scaling_randomization"]["horizontal_bounds"])
        fy = rand_between(*self.config_["objects"]["scaling_randomization"]["vertical_bounds"])
        image = cv2.resize(image, dsize=None, fx=fx, fy=fy, interpolation=cv2.INTER_NEAREST)

        image = self._scale_object_image_to_output_shape(image)

        return image

    def _scale_object_image_to_output_shape(self, image: np.ndarray):
        output_width, output_height = self.config_["io"]["width"], self.config_["io"]["height"]
        min_width = self.config_["composition"]["object_occupancy"]["min_width_percentage"] * output_width
        max_width = self.config_["composition"]["object_occupancy"]["max_width_percentage"] * output_width
        min_height = self.config_["composition"]["object_occupancy"]["min_height_percentage"] * output_height
        max_height = self.config_["composition"]["object_occupancy"]["max_height_percentage"] * output_height

        scale = rand_between(
            max(min_width / image.shape[1], min_height / image.shape[0]),
            min(max_width / image.shape[1], max_height / image.shape[0]),
        )
        image_fitted = cv2.resize(image, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

        return image_fitted


if __name__ == "__main__":
    generator = Generator2D(config_filepath=sys.argv[1])
    generator.generate()
