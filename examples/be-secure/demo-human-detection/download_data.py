import fiftyone
import os

dataset = fiftyone.zoo.load_zoo_dataset(
    "coco-2017",
    split="train",
    # label_types=["detections"],
    classes=["person"],
    max_samples=100,
)

# Export the loaded COCO 'person' subset to a YOLO/Darknet-style directory
export_dir = './data/coco_person_yolo'
os.makedirs(export_dir, exist_ok=True)
print('Exporting dataset to', export_dir)
# Try common FiftyOne YOLO export types; fall back with informative message
try:
    dataset.export(export_dir=export_dir, dataset_type=fiftyone.types.YOLOv4Dataset)
    print('Exported using fiftyone.types.YOLOv4Dataset')
except Exception:
    try:
        dataset.export(export_dir=export_dir, dataset_type=fiftyone.types.YOLOv5Dataset)
        print('Exported using fiftyone.types.YOLOv5Dataset')
    except Exception as e:
        print('Automatic export failed:', e)
        print('If export fails, please export the dataset to YOLO/Darknet format manually using FiftyOne or other tools.')
