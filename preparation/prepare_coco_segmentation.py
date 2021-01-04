import os
import sys
import time
import numpy as np
sys.path.append(os.path.dirname(sys.path[0]))

from PIL import Image
from config import cs_coco_roots
from src.dataset.coco import COCO
from pycocotools.coco import COCO as coco_tools


def main():
    start = time.time()
    root = cs_coco_roots.coco_root
    split = "train"
    year = 2017
    id_in = COCO.train_id_in
    id_out = COCO.train_id_out
    min_size = COCO.min_image_size
    annotation_file = '{}/annotations/instances_{}.json'.format(root, split+str(year))
    images_dir = '{}/{}'.format(root, split+str(year))
    tools = coco_tools(annotation_file)
    save_dir = '{}/annotations/ood_seg_{}'.format(root, split+str(year))
    print("\nPrepare COCO{} {} split for OoD training".format(str(year), split))

    # Names of classes that are excluded - these are Cityscapes classes also available in COCO
    exclude_classes = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'traffic light', 'stop sign']

    # Fetch all image ids that does not include instance from classes defined in "exclude_classes"
    exclude_cat_Ids = tools.getCatIds(catNms=exclude_classes)
    exclude_img_Ids = []
    for cat_Id in exclude_cat_Ids:
        exclude_img_Ids += tools.getImgIds(catIds=cat_Id)
    exclude_img_Ids = set(exclude_img_Ids)
    img_Ids = [int(image[:-4]) for image in os.listdir(images_dir) if int(image[:-4]) not in exclude_img_Ids]

    num_masks = 0
    # Process each image
    print("Ground truth segmentation mask will be saved in:", save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print("Created save directory:", save_dir)
    for i, img_Id in enumerate(img_Ids):
        img = tools.loadImgs(img_Id)[0]
        h, w = img['height'], img['width']

        # Select only images with height and width of at least min_size
        if h >= min_size and w >= min_size:
            ann_Ids = tools.getAnnIds(imgIds=img['id'], iscrowd=None)
            annotations = tools.loadAnns(ann_Ids)

            # Generate binary segmentation mask
            mask = np.ones((h, w), dtype="uint8") * id_in
            for j in range(len(annotations)):
                mask = np.maximum(tools.annToMask(annotations[j])*id_out, mask)

            # Save segmentation mask
            Image.fromarray(mask).save(os.path.join(save_dir, "{:012d}.png".format(img_Id)))
            num_masks += 1
        print("\rImages Processed: {}/{}".format(i + 1, len(img_Ids)), end=' ')
        sys.stdout.flush()

    # Print summary
    print("\nNumber of created segmentation masks with height and width of at least %d pixels:" % min_size, num_masks)
    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("FINISHED {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

if __name__ == '__main__':
    main()
