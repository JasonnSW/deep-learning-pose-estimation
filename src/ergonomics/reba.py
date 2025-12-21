import numpy as np
from ergonomics.angles import angle

def compute_angles(kp):

    def safe_val(val):
        return val if val is not None else 0.0

    def safe_mean(values):
        valid = [v for v in values if v is not None]
        return np.mean(valid) if valid else 0.0
    
    angles = {}

    angles["neck"] = safe_val(angle(kp[5], kp[0], kp[6]))

    shoulder_mid = (kp[5] + kp[6]) / 2
    hip_mid = (kp[11] + kp[12]) / 2
    vertical = hip_mid + np.array([0, 100])
    angles["trunk"] = safe_val(angle(shoulder_mid, hip_mid, vertical))

    left_leg = angle(kp[11], kp[13], kp[15])
    right_leg = angle(kp[12], kp[14], kp[16])
    angles["legs"] = safe_mean([left_leg, right_leg])

    angles["upper_arm"] = safe_mean([
        angle(kp[7], kp[5], shoulder_mid),
        angle(kp[8], kp[6], shoulder_mid)
    ])

    angles["lower_arm"] = safe_mean([
        angle(kp[5], kp[7], kp[9]),
        angle(kp[6], kp[8], kp[10])
    ])

    angles["wrist"] = safe_mean([
        angle(kp[7], kp[9], kp[9] + [1, 0]),
        angle(kp[8], kp[10], kp[10] + [1, 0])
    ])

    return angles


def compute_reba(angles):
    def score_neck(a):
        return 1 if a < 10 else 2 if a < 20 else 3

    def score_trunk(a):
        if a < 5:
            return 1
        elif a < 20:
            return 2
        elif a < 60:
            return 3
        return 4

    def score_legs(a):
        return 1 if a < 30 else 2

    def score_upper_arm(a):
        return 1 if a < 20 else 2 if a < 45 else 3

    def score_lower_arm(a):
        return 1 if 60 <= a <= 100 else 2

    def score_wrist(a):
        return 1 if a < 15 else 2

    A = (
        score_neck(angles["neck"]) +
        score_trunk(angles["trunk"]) +
        score_legs(angles["legs"])
    )

    B = (
        score_upper_arm(angles["upper_arm"]) +
        score_lower_arm(angles["lower_arm"]) +
        score_wrist(angles["wrist"])
    )

    return min(A + B, 15)

def classify_reba(score):
    if score <= 3:
        return "Low"
    elif score <= 7:
        return "Medium"
    elif score <= 10:
        return "High"
    return "Very High"