import json
import sys
from copy import deepcopy
from os.path import splitext

labels_path_raw = sys.argv[1]
labels_raw = json.load(open(labels_path_raw, "r"))
labels_path_raw_split = splitext(labels_path_raw)
validation_ratio = float(sys.argv[2])
validation_n = int(validation_ratio * len(labels_raw["images"]))

labels_training = {"images": labels_raw["images"][validation_n:]}
labels_validation = {"images": labels_raw["images"][:validation_n]}
for labels_dict, suffix in zip([labels_training, labels_validation], ["training", "validation"]):
    labels_dict["categories"] = labels_raw["categories"]
    labels_dict["type"] = labels_raw["type"]

    labels_dict["annotations"] = []
    for image_id_new, image in enumerate(labels_dict["images"], start=1):
        for annotation in labels_raw["annotations"]:
            if annotation["image_id"] == image["id"]:
                annotation = deepcopy(annotation)
                annotation["id"] = len(labels_dict["annotations"]) + 1
                annotation["image_id"] = image_id_new
                labels_dict["annotations"].append(annotation)
        image["id"] = image_id_new

    labels_path = labels_path_raw_split[0] + "_" + suffix + labels_path_raw_split[1]
    json.dump(obj=labels_dict, fp=open(labels_path, "w"))
