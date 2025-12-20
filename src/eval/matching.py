from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

# Returns list of (gt_idx, pred_idx)
def match_people(gt_people, pred_people):
    cost = cdist(
        [kp.flatten() for kp in gt_people],
        [kp.flatten() for kp in pred_people]
    )

    gt_idx, pred_idx = linear_sum_assignment(cost)
    return list(zip(gt_idx, pred_idx))
