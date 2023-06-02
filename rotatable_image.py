import cv2
import numpy as np


def _get_rotated_pixel_coordinate(px, rotation_matrix, intrinsic_matrix):
    point = np.asarray([px[0], px[1], intrinsic_matrix[0, 0]]).reshape(3, 1)
    pt_rotated = np.matmul(rotation_matrix, point)
    px = np.matmul(intrinsic_matrix, pt_rotated / pt_rotated[2]).ravel()[:2]

    return px


class RotatableImage:
    def __init__(self, image_bgra, focal_length=None):
        assert image_bgra.shape[2] == 4

        self.image_bgra_ = image_bgra
        self.focal_length_ = np.linalg.norm(image_bgra.shape[:2]) if focal_length is None else focal_length

        height, width = image_bgra.shape[:2]
        self.image_corners_ = np.asarray([[0, 0], [0, height], [width, 0], [width, height]]).astype(np.float32)

    @property
    def focal_length(self):
        return self.focal_length_

    def set_focal_length(self, focal_length):
        self.focal_length_ = focal_length

    def get_rotated_image(self, rotation_matrix, origin_uv=None):
        origin_uv = [0.5, 0.5] if origin_uv is None else origin_uv
        origin_px = np.asarray(origin_uv) * self.image_bgra_.shape[:2][::-1]

        intrinsic_matrix = np.asarray(
            [[self.focal_length_, 0, origin_px[0]], [0, self.focal_length_, origin_px[1]], [0, 0, 1]]
        )
        corners_rotated = np.asarray(
            [
                _get_rotated_pixel_coordinate(px, rotation_matrix, intrinsic_matrix)
                for px in self.image_corners_ - origin_px
            ]
        )
        corners_rotated -= np.min(corners_rotated, axis=0)

        image_rotated = cv2.warpPerspective(
            src=self.image_bgra_,
            M=cv2.getPerspectiveTransform(self.image_corners_, corners_rotated[:, :2].astype(np.float32)),
            dsize=np.ceil(np.max(corners_rotated, axis=0)).astype(np.int32),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )

        return image_rotated
