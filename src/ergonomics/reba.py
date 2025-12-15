import numpy as np
from ergonomics.angles import angle

def reba_lite_score(points):
    """
    Simplified REBA (upper body only)
    """
    nose = points[0]
    shoulder = points[5]
    hip = points[11]

    neck_angle = angle(nose, shoulder, hip)
    trunk_angle = angle(shoulder, hip, [hip[0], hip[1] + 100])

    score = 0

    # Neck
    if neck_angle is not None:
        if neck_angle < 140:
            score += 2
        elif neck_angle < 160:
            score += 1

    # Trunk
    if trunk_angle is not None:
        if trunk_angle < 140:
            score += 3
        elif trunk_angle < 160:
            score += 2
        else:
            score += 1

    return score


def classify_reba(score):
    if score <= 2:
        return "Low Risk"
    elif score <= 4:
        return "Medium Risk"
    return "High Risk"
