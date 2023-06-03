import json
import os
import pathlib
import sys

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from layered_image import LayeredImage, rand_between
from rotatable_image import RotatableImage


def randomize_hsv(image_bgr, randomization_bounds):
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.int32)
    image_hsv += np.round([rand_between(*bounds) for bounds in randomization_bounds]).astype(np.int32)
    image_hsv[..., 0] = (image_hsv[..., 0] + 180) % 180
    image_hsv[..., 1:] = np.clip(image_hsv[..., 1:], 0, 255)

    image_bgr_randomized = cv2.cvtColor(image_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return image_bgr_randomized


def load_rotatable_images(category_config_filepath: str):
    rotatable_image_dicts, categories = [], []
    supercategory_config = json.load(open(category_config_filepath, "r"))
    data_dir = pathlib.Path(category_config_filepath).parent
    for supercategory, category_config in supercategory_config.items():
        for category_name, image_filenames in category_config.items():
            categories.append({"id": len(categories) + 1, "name": category_name, "supercategory": supercategory})
            for image_filename in image_filenames:
                image = cv2.imread(os.path.join(data_dir, image_filename), cv2.IMREAD_UNCHANGED)
                if not isinstance(image, np.ndarray) or len(image.shape) != 3:
                    continue
                image = image if image.shape[2] == 4 else cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
                rotatable_image_dicts.append({"category_id": categories[-1]["id"], "image": RotatableImage(image)})

    return rotatable_image_dicts, categories


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
        self.objects_, self.categories_ = load_rotatable_images(self.config_["io"]["category_config_filepath"])

    def generate(self):
        metadata = {"type": "instances", "categories": self.categories_, "images": [], "annotations": []}
        if not os.path.exists(self.config_["io"]["output_dir"]):
            os.mkdir(self.config_["io"]["output_dir"])

        unannotated_gen_modulus = int(self.config_["io"]["num_images"] * self.config_["io"]["unannotated_ratio"])
        while len(metadata["images"]) < self.config_["io"]["num_images"]:
            layered_image = LayeredImage(self._generate_background())
            if unannotated_gen_modulus <= 0 or len(metadata["images"]) % unannotated_gen_modulus != 0:
                category_indices_to_add = np.random.choice(
                    len(self.categories_),
                    self.config_["composition"]["layering"]["max_count"],
                    replace=self.config_["composition"]["layering"]["allow_multiple_instances_of_same_category"],
                )
                for category_id_to_add in [self.categories_[index]["id"] for index in category_indices_to_add]:
                    object_index_to_add = np.random.choice(
                        [index for index, obj in enumerate(self.objects_) if obj["category_id"] == category_id_to_add]
                    )
                    object_to_add = self.objects_[object_index_to_add]

                    failed_attempt_count = 0
                    while len(layered_image) < self.config_["composition"]["layering"]["max_count"]:
                        if not layered_image.add_layer_at_random_position(
                            category_id=object_to_add["category_id"],
                            image_bgra=self._randomize_rotatable_image(object_to_add["image"]),
                            visibility_threshold=self.config_["composition"]["layering"]["visibility_threshold"],
                            failed_attempt_limit=self.config_["composition"]["layering"]["failed_attempt_limit"],
                        ):
                            failed_attempt_count += 1
                            if failed_attempt_count == self.config_["composition"]["layering"]["attempt_limit"]:
                                break
                image_ok = len(layered_image) >= self.config_["composition"]["layering"]["min_count"]
                annotations = layered_image.get_annotations() if image_ok else []
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
        json.dump(obj=metadata, fp=open(metadata_filepath, "w"))

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
