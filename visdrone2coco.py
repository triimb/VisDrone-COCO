import os
import json
import shutil
from PIL import Image
from tqdm import tqdm

# Full list of VisDrone categories (including 0 and 11)
class_names = [
    "ignored-regions", "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor", "others"
]

# Create COCO-style categories
categories = [{"id": i, "name": name, "supercategory": "object"} for i, name in enumerate(class_names)]

def convert_visdrone_to_coco(visdrone_root, output_root):
    splits = {
        "train": "VisDrone2019-DET-train",
        "val": "VisDrone2019-DET-val",
        "test": "VisDrone2019-DET-test-dev"
    }

    os.makedirs(os.path.join(output_root, "annotations"), exist_ok=True)

    for split, folder in splits.items():
        print(f"\n Processing split: {split}")
        image_dir = os.path.join(visdrone_root, folder, "images")
        annot_dir = os.path.join(visdrone_root, folder, "annotations")
        output_img_dir = os.path.join(output_root, split)
        os.makedirs(output_img_dir, exist_ok=True)

        images = []
        annotations = []
        ann_id = 1
        img_id = 1

        for img_name in tqdm(sorted(os.listdir(image_dir))):
            if not img_name.lower().endswith(".jpg"):
                continue

            shutil.copy(os.path.join(image_dir, img_name), os.path.join(output_img_dir, img_name))

            width, height = Image.open(os.path.join(image_dir, img_name)).size
            images.append({
                "id": img_id,
                "file_name": img_name,
                "width": width,
                "height": height
            })

            txt_file = os.path.join(annot_dir, img_name.replace(".jpg", ".txt"))
            if os.path.exists(txt_file):
                with open(txt_file, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line or ',' not in line:
                            continue

                        parts = line.split(",")
                        if len(parts) < 8:
                            continue

                        try:
                            x, y, w, h, score, cat_id, trunc, occ = map(int, parts)
                        except ValueError:
                            continue

                        if w <= 0 or h <= 0 or cat_id < 0 or cat_id > 11:
                            continue  # Invalid values

                        annotations.append({
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": cat_id,  # keep original 0-11 category_id
                            "bbox": [x, y, w, h],
                            "area": w * h,
                            "iscrowd": 0
                        })
                        ann_id += 1
            img_id += 1

        out_json = {
            "info": {
                "description": f"VisDrone converted to COCO - {split}",
                "version": "1.0",
                "year": 2025
            },
            "licenses": [],
            "images": images,
            "annotations": annotations,
            "categories": categories
        }

        with open(os.path.join(output_root, "annotations", f"instances_{split}.json"), "w") as f:
            json.dump(out_json, f, indent=2)

    print("\nAll splits processed successfully.")

# Example usage
if __name__ == "__main__":
    visdrone_path = "VisDrone/"
    coco_output_path = "VisDrone-COCO/"
    convert_visdrone_to_coco(visdrone_path, coco_output_path)
