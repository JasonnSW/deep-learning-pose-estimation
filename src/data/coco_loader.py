import os
import numpy as np
from pycocotools.coco import COCO

class CocoPoseDataset:
    def __init__(self, ann_path: str, image_dir: str):
        self.coco = COCO(ann_path)
        self.image_dir = image_dir

    # Returns a list of image IDs that contain persons
    def get_valid_image_ids(self, max_images=None):
        all_ids = self.coco.getImgIds()
        
        if max_images:
            all_ids = all_ids[:max_images]

        valid_ids = []
        for img_id in all_ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=[1])
            if len(ann_ids) > 0:
                valid_ids.append(img_id)
                
        return valid_ids

    # Yields (image_path, ground_truth) for the given IDs.
    def get_samples(self, img_ids):
        for img_id in img_ids:
            img_info = self.coco.loadImgs(img_id)[0]
            img_path = os.path.join(self.image_dir, img_info["file_name"])

            ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=[1])
            anns = self.coco.loadAnns(ann_ids)

            gt_people = []
            for ann in anns:
                if ann["num_keypoints"] == 0:
                    continue
                kp = np.array(ann["keypoints"]).reshape(17, 3)[:, :2]
                gt_people.append(kp)

            if len(gt_people) > 0:
                yield img_path, gt_people