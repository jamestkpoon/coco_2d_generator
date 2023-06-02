import glob
import json
import os
import pathlib
import sys

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from layered_image import LayeredImage, rand_between
from rotatable_image import RotatableImage


def randomize_hsv(image_bgr, perturbation_bounds):
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.int32)
    image_hsv += np.round([rand_between(*bounds) for bounds in perturbation_bounds]).astype(np.int32)
    image_hsv = np.clip(image_hsv, a_min=0, a_max=255).astype(np.uint8)

    image_bgr_randomized = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)

    return image_bgr_randomized


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


def randomly_rotate_image(rotatable_image: RotatableImage, euler_bounds_xyz, origin_uv_bounds):
    euler_angles = [rand_between(*bounds) for bounds in euler_bounds_xyz]
    rotation_matrix = Rotation.from_euler("xyz", euler_angles).as_matrix()

    origin_uv = [rand_between(*bounds) for bounds in origin_uv_bounds]
    rotated_image = rotatable_image.get_rotated_image(rotation_matrix, origin_uv)

    return rotated_image


class Generator2D:
    def __init__(self, config_filepath: str):
        self.config_ = json.load(open(config_filepath, "r"))
        self.rotatable_images_ = load_rotatable_images(self.config_["io"]["template_dir"])

    def generate(self):
        labels = sorted(self.rotatable_images_.keys())
        metadata = {
            "type": "instances",
            "categories": [{"supercategory": "none", "id": index, "name": label} for index, label in enumerate(labels)],
            "images": [],
            "annotations": [],
        }

        unannotated_gen_modulus = int(self.config_["io"]["num_images"] * self.config_["io"]["unannotated_ratio"])
        if not os.path.exists(self.config_["io"]["output_dir"]):
            os.mkdir(self.config_["io"]["output_dir"])

        while len(metadata["images"]) < self.config_["io"]["num_images"]:
            layered_image = LayeredImage(self._generate_background())
            if unannotated_gen_modulus <= 0 or len(metadata["images"]) % unannotated_gen_modulus != 0:
                label_indices_to_add = np.random.choice(
                    len(labels),
                    self.config_["composition"]["layering"]["max_count"],
                    replace=self.config_["composition"]["layering"]["allow_multiple_instances_of_same_object"],
                )
                for label_index_to_add in label_indices_to_add:
                    rotatable_image = self.rotatable_images_[labels[label_index_to_add]]
                    attempt_count = 0
                    while len(layered_image) < self.config_["composition"]["layering"]["max_count"]:
                        if not layered_image.add_layer(
                            category_id=int(label_index_to_add),
                            image_bgra=self._randomize_rotatable_image(rotatable_image),
                            visibility_threshold=self.config_["composition"]["layering"]["visibility_threshold"],
                            attempt_limit=self.config_["composition"]["layering"]["attempt_limit"],
                        ):
                            attempt_count += 1
                            if attempt_count == self.config_["composition"]["layering"]["attempt_limit"]:
                                break

                annotations = layered_image.get_annotations()
                image_ok = len(annotations) >= self.config_["composition"]["layering"]["min_count"]
            else:
                image_ok, annotations = True, []

            if image_ok:
                image_metadata = {
                    "id": (image_id := len(metadata["images"]) + 1),
                    "file_name": "{}{}".format(image_id, self.config_["io"]["extension"]),
                    "width": self.config_["io"]["width"],
                    "height": self.config_["io"]["height"],
                }
                metadata["images"].append(image_metadata)
                image_filepath = os.path.join(self.config_["io"]["output_dir"], image_metadata["file_name"])
                cv2.imwrite(image_filepath, layered_image.composite)

                for annotation in annotations:
                    annotation["image_id"] = image_metadata["id"]
                    annotation["id"] = len(metadata["annotations"]) + 1
                    metadata["annotations"].append(annotation)

        metadata_filepath = os.path.join(self.config_["io"]["output_dir"], "metadata.json")
        json.dump(obj=metadata, fp=open(metadata_filepath, "w"), indent=4)

    def _generate_background(self):
        return self._randomize_hsv(
            generate_background(
                **self.config_["background"],
                output_width=self.config_["io"]["width"],
                output_height=self.config_["io"]["height"],
            )
        )

    def _randomize_rotatable_image(self, rotatable_image: RotatableImage):
        image = randomly_rotate_image(rotatable_image, **self.config_["objects"]["rotation_randomization"])

        image[..., :3] = self._randomize_hsv(image[..., :3])

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

    def _randomize_hsv(self, image_bgr):
        return randomize_hsv(image_bgr, self.config_["hsv_randomization_bounds"])


if __name__ == "__main__":
    generator = Generator2D(config_filepath=sys.argv[1])
    generator.generate()
