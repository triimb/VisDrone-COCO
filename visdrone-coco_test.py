from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from PIL import Image

def visualize_coco_sample(json_path, image_dir, num_images=5):
    coco = COCO(json_path)
    cat_ids = coco.getCatIds()
    img_ids = coco.getImgIds()

    for i in range(min(num_images, len(img_ids))):
        img_info = coco.loadImgs(img_ids[i])[0]
        img_path = os.path.join(image_dir, img_info['file_name'])
        img = Image.open(img_path)

        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        ax = plt.gca()

        ann_ids = coco.getAnnIds(imgIds=img_info['id'], catIds=cat_ids, iscrowd=None)
        anns = coco.loadAnns(ann_ids)

        for ann in anns:
            x, y, w, h = ann['bbox']
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            cat_name = coco.loadCats(ann['category_id'])[0]['name']
            plt.text(x, y - 5, cat_name, color='yellow', fontsize=12, backgroundcolor='black')

        plt.title(img_info['file_name'])
        plt.axis('off')
        plt.show()

# Example usage
json_file = "VisDrone-COCO/annotations/instances_val.json"
images_folder = "VisDrone-COCO/val"
visualize_coco_sample(json_file, images_folder, num_images=3)
