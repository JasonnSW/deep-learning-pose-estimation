import os
import json
from datetime import datetime
from ultralytics import YOLO

from data.coco_loader import CocoPoseDataset
from eval.evaluate import evaluate
from utils.logging import setup_logger
from viz.plots import generate_plots


def main():
    # ------------------------------------------------------------------
    # Run setup
    # ------------------------------------------------------------------
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    base_dir = f"results/runs/{run_id}"
    log_dir = f"{base_dir}/logs"
    metrics_dir = f"{base_dir}/metrics"
    viz_dir = f"{base_dir}/visualizations"

    for d in [log_dir, metrics_dir, viz_dir]:
        os.makedirs(d, exist_ok=True)

    logger = setup_logger(f"{log_dir}/run.log")

    logger.info("========== REBA Evaluation Started ==========")

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    dataset = CocoPoseDataset(
        ann_path="datasets/coco/annotations/person_keypoints_val2017.json",
        image_dir="datasets/coco/val2017"
    )

    logger.info("Scanning dataset for valid images...")
    valid_ids = dataset.get_valid_image_ids()
    valid_ids = valid_ids[101:150] # delete later
    
    used_images = len(valid_ids)
    total_images = len(dataset.coco.getImgIds())

    logger.info(f"Total COCO images: {total_images}")
    logger.info(f"Images with persons used: {used_images}")
    print(f"Images to be evaluated: {used_images}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = YOLO("yolov8s-pose.pt")

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    samples = dataset.get_samples(valid_ids)
    results = evaluate(model, samples)

    persons_evaluated = results["persons"]
    mae = results["mae"]
    risk_accuracy = results["risk_accuracy"]
    risk_diff = results["risk_diff"]

    print(f"Persons evaluated: {persons_evaluated}")
    print(f"Mean Absolute Error (REBA): {mae}")

    print("\nRisk level mismatches:")
    for k, v in risk_diff.items():
        print(f"{k} → {v}")

    print(f"\nRisk-level accuracy: {risk_accuracy}")

    # ------------------------------------------------------------------
    # Save metrics
    # ------------------------------------------------------------------
    with open(f"{metrics_dir}/summary.json", "w") as f:
        json.dump(
            {
                "total_images": total_images,
                "images_used": used_images,
                "persons_evaluated": persons_evaluated,
                "mae": mae,
                "risk_accuracy": risk_accuracy,
            },
            f,
            indent=2
        )

    with open(f"{metrics_dir}/risk_confusion.json", "w") as f:
        json.dump(
            {str(k): v for k, v in risk_diff.items()},
            f,
            indent=2
        )

    # ------------------------------------------------------------------
    # Visualizations
    # ------------------------------------------------------------------
    print("\n[INFO] Generating plots...")
    generate_plots(results, viz_dir)

    # ------------------------------------------------------------------
    # Metadata 
    # ------------------------------------------------------------------
    with open(f"{base_dir}/metadata.json", "w") as f:
        json.dump(
            {
                "run_id": run_id,
                "model": "yolov8s-pose.pt",
                "confidence_threshold": 0.3,
                "dataset": "COCO val2017",
            },
            f,
            indent=2
        )

    logger.info("========== Evaluation Completed ==========")
    logger.info(f"Persons evaluated: {persons_evaluated}")
    logger.info(f"MAE: {mae}")
    logger.info(f"Risk accuracy: {risk_accuracy}")


if __name__ == "__main__":
    main()
