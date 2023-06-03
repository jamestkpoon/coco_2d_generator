import cv2
import numpy as np


def rand_between(lower, upper):
    return lower + np.random.rand() * (upper - lower)


def _layer_properties_to_annotation(layer_properties: dict):
    contour = cv2.findContours(
        layer_properties["mask_whole"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )[0][0]

    return {
        "category_id": layer_properties["category_id"],
        "segmentation": [contour.ravel().tolist()],
        "area": cv2.contourArea(contour),
        "bbox": cv2.boundingRect(contour),
        "iscrowd": 0,
    }


class LayeredImage:
    def __init__(self, background: np.ndarray):
        self.canvas_, self.layer_properties_ = background, []

    @property
    def composite(self):
        return self.canvas_

    def __len__(self):
        return len(self.layer_properties_)

    def get_annotations(self):
        return [_layer_properties_to_annotation(layer_properties) for layer_properties in self.layer_properties_]

    def add_layer(
        self,
        category_id: int,
        image_bgra: np.ndarray,
        canvas_slice: tuple,
        image_slice: tuple,
        visibility_threshold: float,
    ):
        image, image_mask = image_bgra[..., :3], image_bgra[..., 3] != 0
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
            return False

        for layer_properties, mask_visible_new in zip(self.layer_properties_, other_masks_visible_new):
            layer_properties["mask_visible"] = mask_visible_new
        self.layer_properties_.append(
            {
                "category_id": category_id,
                "mask_whole": canvas_mask,
                "mask_visible": canvas_mask,
                "mask_visible_threshold": int(np.count_nonzero(canvas_mask[canvas_slice]) * visibility_threshold),
            }
        )
        self.canvas_[canvas_slice][image_mask[image_slice]] = image[image_slice][image_mask[image_slice]]

        return True

    def add_layer_at_random_position(
        self, category_id: int, image_bgra: np.ndarray, visibility_threshold: float, failed_attempt_limit: int
    ):
        failed_attempt_count = 0
        while failed_attempt_count < failed_attempt_limit:
            canvas_slice, image_slice = self._get_random_image_slice(image_bgra.shape, visibility_threshold)
            if self.add_layer(category_id, image_bgra, canvas_slice, image_slice, visibility_threshold):
                return True

            failed_attempt_count += 1

        return False

    def _get_random_image_slice(self, image_shape: tuple[int], visibility_threshold: float):
        image_center_x_offset_limit = int(np.round(image_shape[1] * (visibility_threshold - 0.5)))
        image_center_y_offset_limit = int(np.round(image_shape[0] * (visibility_threshold - 0.5)))
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
        canvas_xmax = min(self.canvas_.shape[1], canvas_xmin_raw + image_shape[1])
        canvas_ymax = min(self.canvas_.shape[0], canvas_ymin_raw + image_shape[0])
        image_xmin = half_image_shape[1] - (image_center_x_offset - canvas_xmin)
        image_ymin = half_image_shape[0] - (image_center_y_offset - canvas_ymin)
        image_xmax = image_xmin + (canvas_xmax - canvas_xmin)
        image_ymax = image_ymin + (canvas_ymax - canvas_ymin)

        canvas_slice = np.index_exp[canvas_ymin:canvas_ymax, canvas_xmin:canvas_xmax]
        image_slice = np.index_exp[image_ymin:image_ymax, image_xmin:image_xmax]

        return canvas_slice, image_slice
