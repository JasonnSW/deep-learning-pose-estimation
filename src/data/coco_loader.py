import json

class CocoKeypointsLoader:
    def __init__(self, ann_path):
        with open(ann_path, "r") as f:
            self.data = json.load(f)

    def get_person_keypoints(self):
        keypoints_list = []
        for ann in self.data["annotations"]:
            if ann["num_keypoints"] > 0:
                kp = ann["keypoints"] 
                keypoints_list.append(kp)
        return keypoints_list
