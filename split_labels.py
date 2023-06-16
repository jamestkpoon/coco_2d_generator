import json
import sys
from copy import deepcopy
from os.path import splitext

labels_path_raw = sys.argv[1]
labels_raw = json.load(open(labels_path_raw, "r"))
labels_path_raw_split = splitext(labels_path_raw)
validation_ratio = float(sys.argv[2])
validation_n = int(validation_ratio * len(labels_raw["images"]))

for images, suffix in zip(
    [labels_raw["images"][validation_n:], labels_raw["images"][:validation_n]], ["training", "validation"]
):
    annotations = {image["id"]: [] for image in images}
    for annotation in labels_raw["annotations"]:
        annotations_list = annotations.get(annotation["image_id"])
        if isinstance(annotations_list, list):
            annotations_list.append(annotation)

    labels = {"type": labels_raw["type"], "categories": labels_raw["categories"], "images": images, "annotations": []}
    for image_id_new, image in enumerate(images, start=1):
        for annotation in annotations.get(image["id"], []):
            annotation = deepcopy(annotation)
            annotation["id"] = len(labels["annotations"]) + 1
            annotation["image_id"] = image_id_new
            labels["annotations"].append(annotation)
        image["id"] = image_id_new

    labels_path = labels_path_raw_split[0] + "_" + suffix + labels_path_raw_split[1]
    json.dump(obj=labels, fp=open(labels_path, "w"))
