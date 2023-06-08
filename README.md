# coco2d_generator

Generates datasets for object instance detection, in [COCO format](https://towardsdatascience.com/how-to-work-with-object-detection-datasets-in-coco-format-9bf4fb5848a4).

Requirements: cv2, numpy, scipy
```commandline
pip3 install -r requirements.txt 
```

Usage:
```commandline
python3 generator2D.py /path/to/config.json
```

## Category template configuration
Category configuration is expected to be a .json structured as follows:
```commandline
{
  "supercategory1_name": {
    "category1_name": ["template1.png", "template2.png", ...],
    "category2_name": ["template3.png", "template4.png", ...],
    ...
  },
  "supercategory2_name": {
    "category4_name": ["template15.png", "template16.png", ...],
    ...
  },
  ...
}
```
where template1.png, template2.png, ..., templateN.png are names of image files in the same folder.
.png format is recommended for its opacity channel.

## config.json
Please refer to the example ./config.json.
```commandline
{
  "io": {
    "category_config_filepath": full path to category_config.json (str),
    "output_dir": full path to output folder, will be created if it does not exist (str),
    "extension": output image file extension, e.g. ".jpg", ".png" (str),
    "num_images": number of images to generate (int),
    "unannotated_ratio": ratio of generated images that will have no objects, to possibly reduce false positives (float),
    "width": output image width (int),
    "height": output image height (int)
  },
  "hsv_randomization_bounds": lower and upper bounds for randomization in HSV space. Applied separately to background and each object template (list[list[int]]),
  "segmentation_approxpoly_eps": contour perimeter coefficient for segmentation simplification via cv2.approxPolyDP (float),
  "background": {
    "min_random_width": min random background width, to be resized to full output size (int),
    "max_random_width": max random background width, to be resized to full output size (int),
    "min_random_height": min random background height, to be resized to full output size (int),
    "max_random_height": max random background height, to be resized to full output size (int)
  },
  "objects": {
    "rotation_randomization": {
      "euler_bounds_xyz": lower and upper bounds for template rotation along x,y,z axes in radians (list[list[float]]),
      "origin_uv_bounds": lower and upper bounds for rotation origin (recomended between 0 and 1), along x,y axes (list[list[float]])
    },
    "scaling_randomization": {
      "horizontal_bounds": lower and upper bounds for template scaling along the x axis (list[float]),
      "vertical_bounds": lower and upper bounds for template scaling along the y axis (list[float])
    }
  },
  "composition": {
    "object_occupancy": {
      "min_width_percentage": min x-axis ratio of scaled template to output image shape (float),
      "min_height_percentage": max x-axis ratio of scaled template to output image shape (float),
      "max_width_percentage": min y-axis ratio of scaled template to output image shape (float),
      "max_height_percentage": max y-axis ratio of scaled template to output image shape (float)
    },
    "layering": {
      "min_count": min objects per output image (int),
      "max_count": max objects per output image (int),
      "allow_multiple_instances_of_same_category": used as replace kwarg in np.random.choice for selecting categories to draw from (boolean,
      "visibility_threshold": visibility ratio threshold of each object for the final image to be valid (float),
      "failed_attempt_limit": cap of random object placement iterations (int)
    }
  }
}
```