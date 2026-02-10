import fiftyone

dataset = fiftyone.zoo.load_zoo_dataset(
    "coco-2017",
    split="train",
    # label_types=["detections"],
    classes=["person"],
    max_samples=100,
)
