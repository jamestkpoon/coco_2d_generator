import json
import sys
from os.path import splitext

labels_path_raw = sys.argv[1]
labels_raw = json.load(open(labels_path_raw, "r"))
labels_path_raw_split = splitext(labels_path_raw)
validation_ratio = float(sys.argv[2])
validation_n = int(validation_ratio * len(labels_raw["images"]))

image_annotations = {image["id"]: [] for image in labels_raw["images"]}
for annotation in labels_raw["annotations"]:
    image_annotations[annotation["image_id"]].append(annotation)

image_sets = {"training": labels_raw["images"][validation_n:], "validation": labels_raw["images"][:validation_n]}
for set_name, images in image_sets.items():
    labels = {"type": labels_raw["type"], "categories": labels_raw["categories"], "images": images, "annotations": []}
    for image_id_new, image in enumerate(images, start=1):
        for annotation in image_annotations[image["id"]]:
            annotation["id"] = len(labels["annotations"]) + 1
            annotation["image_id"] = image_id_new
            labels["annotations"].append(annotation)
        image["id"] = image_id_new

    labels_path = labels_path_raw_split[0] + "_" + set_name + labels_path_raw_split[1]
    json.dump(obj=labels, fp=open(labels_path, "w"))
