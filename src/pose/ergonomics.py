def detect_pose_state(points, box):
    """
    Robust pose state detection using bounding box shape
    """

    x1, y1, x2, y2 = box
    h = y2 - y1
    w = x2 - x1

    if w == 0 or h == 0:
        return "Unknown"

    aspect_ratio = h / w

    # Tall box -> Standing
    if aspect_ratio > 1.2:
        return "Standing"

    # Short box -> Sitting
    return "Sitting"


def classify_pose(pose_state):
    if pose_state == "Sitting":
        return "Ergonomic"
    if pose_state == "Standing":
        return "Non-Ergonomic"
    return "Unknown"


def is_valid_pose(points):
    required = [11, 13, 15]  
    for idx in required:
        x, y = points[idx]
        if x == 0 or y == 0:
            return False
    return True

