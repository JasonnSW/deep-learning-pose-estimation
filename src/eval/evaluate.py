import numpy as np
from collections import Counter
from ergonomics.reba import compute_angles, compute_reba, classify_reba
from eval.matching import match_people

def evaluate(model, samples, conf=0.3):
    errors = []
    gt_scores = []
    pred_scores = []
    gt_levels = []
    pred_levels = []
    confidences = []

    risk_diff = Counter()
    persons_evaluated = 0
    images_used = 0

    for img_path, gt_people in samples:
        result = model(img_path, conf=conf)[0]

        if result.keypoints is None:
            continue

        pred_people = result.keypoints.xy.cpu().numpy()
        pred_confs = result.keypoints.conf.cpu().numpy()

        if len(pred_people) == 0:
            continue

        images_used += 1
        matches = match_people(gt_people, pred_people)

        for g, p in matches:
            gt_kp = gt_people[g]
            pred_kp = pred_people[p]
            mean_conf = float(pred_confs[p].mean())

            reba_gt = compute_reba(compute_angles(gt_kp))
            reba_pred = compute_reba(compute_angles(pred_kp))

            err = abs(reba_gt - reba_pred)

            errors.append(err)
            gt_scores.append(reba_gt)
            pred_scores.append(reba_pred)
            gt_levels.append(classify_reba(reba_gt))
            pred_levels.append(classify_reba(reba_pred))
            confidences.append(mean_conf)

            persons_evaluated += 1

            if classify_reba(reba_gt) != classify_reba(reba_pred):
                risk_diff[(classify_reba(reba_gt), classify_reba(reba_pred))] += 1

    return {
        "images_used": images_used,
        "persons": persons_evaluated,

        "errors": errors,
        "mae": float(np.mean(errors)) if errors else 0.0,
        "risk_diff": risk_diff,
        "risk_accuracy": 1 - sum(risk_diff.values()) / len(errors) if errors else 0.0,

        "gt_scores": gt_scores,
        "pred_scores": pred_scores,
        "gt_levels": gt_levels,
        "pred_levels": pred_levels,
        "confidences": confidences
    }
